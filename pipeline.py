from functools import partial
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

_STAGE_AXIS = "stage"


def _to_microbatches(input: ArrayLike, num_microbatches: int = 1):
  # The data batch needs to be in shape of (NUM_MICROBATCHES, MICROBATCH_SIZE, NUM_FEATURES)
  # The first dimension will be sharded across the stages for the sake of saving memories.
  # If we have 8 microbatches and 4 stages, the microbatches will be distributed as,
  #   Device 1 (Stage 1): 0, 1
  #   Device 2 (Stage 2): 2, 3
  #   Device 3 (Stage 3): 4, 5
  #   Device 4 (Stage 4): 6, 7
  #   (Assuming each stage contains only 1 device.)
  #
  # This feature is called "stream_io".
  global_batch_size = input.shape[0]
  staged_batch = input.reshape(
    num_microbatches, global_batch_size // num_microbatches, *input.shape[1:]
  )
  return staged_batch


def _rotate_left(x: ArrayLike, num_stages: int = 1):
  """Rotates the global buffer along its first axis to the left.

  This version only works inside a shard_map call. For instance,
  The global array is [0, 1, 2, 3, 4, 5, 6, 7] and sharded 4-way on the
  first dimension. The distributed tensor will become,
  [0, 1]                                 [2, 1]         [1, 2]
  [2, 3] Rotate-left the first column >> [4, 3] roll >> [3, 4]
  [4, 5]                                 [6, 5]         [5, 6]
  [6, 7]                                 [0, 7]         [7, 0]
  """
  if num_stages == 1:
    return x
  xs = list(jnp.unstack(x))
  left = [(i, (i - 1) % num_stages) for i in range(num_stages)]
  xs[0] = jax.lax.ppermute(xs[0], _STAGE_AXIS, left)
  x = jnp.stack(xs)
  return jnp.roll(x, -1, axis=0)


def _rotate_state_to_right(x: ArrayLike, num_stages: int = 1):
  """Rotates the state buffer along its second axis to the right.

  This version only works inside a shard_map call. For instance,
  The global state array is sharded 4-way as,
  [0, 4]                 [3, 7]                     [7, 3]
  [1, 5] Rotate-right >> [0, 4] roll first stage >> [0, 4]
  [2, 6]                 [1, 5]                     [1, 5]
  [3, 7]                 [2, 6]                     [2, 6]
  """
  if num_stages == 1:
    return x
  right = [(i, (i + 1) % num_stages) for i in range(num_stages)]
  x = jax.lax.ppermute(x, _STAGE_AXIS, right)
  stage_index = jax.lax.axis_index(_STAGE_AXIS)
  return jnp.where(stage_index == 0, jnp.roll(x, -1, axis=0), x)


def gpipe_spmd_pipeline(
  stage_fn: Callable,
  num_stages: int,
  num_microbatches: int,
  circular_repeats: int = 1,
  mesh: jax.sharding.Mesh | None = None,
  microbatch_sharding: P | None = None,
):
  """Transforms the stage_fn into a GPipe-style SPMD-pipelined forward function."""
  stage_init, stage_apply = hk.transform(stage_fn)
  auto_axes = {name for name in mesh.axis_names if name != _STAGE_AXIS}
  num_repeated_stages = num_stages * circular_repeats
  m_sharding = (
    NamedSharding(mesh, P(_STAGE_AXIS, *microbatch_sharding))
    if microbatch_sharding
    else None
  )

  def _per_stage_params_sharding(input):
    param_shape = jax.eval_shape(stage_init, jax.random.key(0), input)
    return jax.tree.map(lambda _: P(_STAGE_AXIS), param_shape)

  def pipelined_init_fn(staged_keys, staged_inputs):
    p_sharding = _per_stage_params_sharding(staged_inputs[0])

    @partial(
      shard_map,
      mesh=mesh,
      in_specs=(P(_STAGE_AXIS), P(_STAGE_AXIS)),
      out_specs=p_sharding,
      check_rep=False,
      auto=frozenset(auto_axes),
    )
    def _init(keys, staged_inputs):
      return jax.vmap(stage_init, in_axes=(0, None))(keys[0], staged_inputs[0])

    return _init(staged_keys, staged_inputs)

  def pipelined_apply_fn(staged_params, staged_keys, staged_inputs):
    p_sharding = _per_stage_params_sharding(staged_inputs[0])

    @partial(
      shard_map,
      mesh=mesh,
      in_specs=(p_sharding, P(_STAGE_AXIS), P(_STAGE_AXIS)),
      # Fixes me.
      out_specs=P(_STAGE_AXIS),
      check_rep=False,
      auto=frozenset(auto_axes),
    )
    def _apply(staged_params, keys, staged_inputs):
      """The function arranges pipelined computations in a single stage.

      Args:
          staged_params: A pytree with parameter leaves. The leading axis of each
            leaf spans across circular repeats partitioned to this stage.
          keys: Same keys for initializing each circular repeat.
          staged_inputs: microbatches partitioned to this stage, whose shapes are
            (microbatch_per_stage, microbatch_size, ...)
      """
      stage_index = jax.lax.axis_index(_STAGE_AXIS)
      # TODO(yunlongl): Supports non-divisible num microbatches by adding paddings.
      num_mbs_per_stage = num_microbatches // num_stages
      assert num_microbatches % num_stages == 0

      # Flattens keys.
      keys = keys.reshape(-1)

      #  ==> Shape (microbatch_per_stage, microbatch_size, ...)
      #  For safety purpose, we pollute the garbage values with nans.
      state = jnp.zeros_like(staged_inputs, dtype=staged_inputs.dtype) * jnp.nan

      # Prepare output
      # After SPMD partitioning, input should have a shape of
      #   (NUM_MICROBATCHES_PER_STAGE, MB_SIZE, NUM_FEATURES)
      output_shape_per_microbatch = jax.eval_shape(
        stage_apply,
        jax.tree.map(lambda x: x[0], staged_params),
        keys[0],
        staged_inputs[0],
      )
      output = jnp.zeros(
        (num_mbs_per_stage, *output_shape_per_microbatch.shape),
        dtype=staged_inputs.dtype,
      )
      output_start_idx = (circular_repeats - 1) * num_microbatches + num_stages - 1

      # TODO: Adds a stack layer version after we resolve the perf issue with DS.
      # We need run num_microbatches * CR in order to process all microbatches in
      # stage 0 and we need an extra num_stages - 1 to push through all
      # in-flight activations.
      for i in range(num_microbatches * circular_repeats + num_stages - 1):
        # Parameters for the current virtual stage.
        cr_idx = jnp.minimum(
          (i - stage_index) // num_microbatches, circular_repeats - 1
        )
        cr_params = jax.tree.map(lambda x, cr_idx=cr_idx: x[cr_idx], staged_params)

        # Input for this virtual stage (a repeat inside a stage).
        # Only stage 0 picks up from the first position of the microbatches as input
        # because we always rotate all the staged inputs after one virtual stage.
        # Other stages always pick up activations or garbage values.
        first_stage_input = jnp.where(i < num_microbatches, staged_inputs[0], state[0])
        state = state.at[0].set(
          jnp.where(stage_index == 0, first_stage_input, state[0])
        )

        # Run stage function for this virtual stage.
        state = state.at[0].set(stage_apply(cr_params, keys[cr_idx], state[0]))

        # Write and rotate output before we rotate states.
        if i >= output_start_idx:
          # Write the output to the last spot and we rotate left NUM_MICROBATCHES time.
          output = output.at[num_mbs_per_stage - 1].set(
            jnp.where(
              stage_index == num_stages - 1, state[0], output[num_mbs_per_stage - 1]
            )
          )
          if i < output_start_idx + num_microbatches - 1:
            # No need to rotate for the last write.
            output = _rotate_left(output, num_stages=num_stages)

        # Rotate input
        if i < num_microbatches * circular_repeats:
          # After this amount of rotating, the stage 0 starts to enter
          # the bubble zone.
          staged_inputs = _rotate_left(staged_inputs, num_stages=num_stages)
        # Rotate state
        state = _rotate_state_to_right(state, num_stages=num_stages)

      return output

    return _apply(staged_params, staged_keys, staged_inputs)

  def pipelined_fn(inputs):
    staged_inputs = jax.tree.map(
      partial(_to_microbatches, num_microbatches=num_microbatches), inputs
    )
    if m_sharding is not None:
      staged_inputs = jax.lax.with_sharding_constraint(staged_inputs, m_sharding)
    # Reorder keys to follow the circular repeat semantics.
    #  For instance, (0, 1, 2, 3) ==> (0, 2, 1, 3), if we have 2 stages and 2 repeat.
    #    Device 0: 0     2
    #    Device 1:    1     3
    staged_keys = (
      hk.next_rng_keys(num_repeated_stages).reshape(-1, num_stages).transpose()
    )
    lifted_init = hk.transparent_lift(pipelined_init_fn)
    # The final shape of each leave of the pytree will be (NUM_STAGE * CIRCULAR_REPEAT, ...)
    # After we shard the params with P('stage'), each stage will receive a circular_repeats
    # number of sets. In the above example, stage 0 will receive [params_0, params_2].
    stage_params = lifted_init(staged_keys, staged_inputs)
    outputs = pipelined_apply_fn(stage_params, staged_keys, staged_inputs)
    return jax.tree.map(lambda x, ref: x.reshape(ref.shape), outputs, inputs)

  return pipelined_fn
