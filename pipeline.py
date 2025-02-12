from collections import namedtuple
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
  global_batch_size = input.shape[0]
  staged_batch = input.reshape(
      num_microbatches, global_batch_size // num_microbatches, *input.shape[1:]
  )
  return staged_batch


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
  full_perm = [(i, (i + 1) % num_stages) for i in range(num_stages)]
  # Manually decomposing collective permute to avoid cycles.
  stage_zero_recv = jax.lax.ppermute(x, _STAGE_AXIS, [full_perm[-1]])
  other_stages_recv = jax.lax.ppermute(x, _STAGE_AXIS, full_perm[:-1])
  stage_index = jax.lax.axis_index(_STAGE_AXIS)
  return jnp.where(stage_index == 0,
                   jnp.roll(stage_zero_recv, -1, axis=0),
                   other_stages_recv)


# Args we carried to the next iteration of the layer scan of the pipeline_fn.
PipelineCarry = namedtuple(
    "PipelineCarry", [
        # Intermediate activations
        "states",
        # Outputs
        "outputs",
        # Loop index
        "i"
    ]
)


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
  # Currently, this sharding will distribute all microbatches to
  # all stages. Ideally, we can stream in inputs from pinned memory.
  m_sharding = (
      NamedSharding(mesh, P(None, *microbatch_sharding))
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
        in_specs=(P(_STAGE_AXIS), P(None)),
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
        in_specs=(p_sharding, P(_STAGE_AXIS), P(None)),
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
            (num_microbatches, microbatch_size, ...)
      """
      stage_index = jax.lax.axis_index(_STAGE_AXIS)
      # Flattens keys.
      keys = keys.reshape(-1)

      # Prepare output
      # After SPMD partitioning, input should have a shape of
      #   (NUM_MICROBATCHES_PER_STAGE, MB_SIZE, NUM_FEATURES)
      output_shape_per_microbatch = jax.eval_shape(
          stage_apply,
          jax.tree.map(lambda x: x[0], staged_params),
          keys[0],
          staged_inputs[0],
      )
      #  ==> Shape (num_microbatch / num_stages, microbatch_size, ...)
      #  For safety purpose, we pollute the garbage values with nans.
      states = jnp.zeros(
          (num_microbatches // num_stages, *output_shape_per_microbatch.shape),
          dtype=staged_inputs.dtype) * jnp.nan
      # Here assume only one output tensor.
      outputs = jnp.zeros(
          (num_microbatches, *output_shape_per_microbatch.shape),
          dtype=staged_inputs.dtype,
      )
      output_start_idx = (circular_repeats - 1) * num_microbatches + num_stages - 1

      # We need run num_microbatches * CR in order to process all microbatches in
      # stage 0 and we need another num_stages - 1 to push through all
      # in-flight activations.
      num_iterations = num_microbatches * circular_repeats + num_stages - 1

      def _compute_fn(c: PipelineCarry):
        states, outputs, i = c.states, c.outputs, c.i
        cr_idx = jnp.minimum(
            (i - stage_index) // num_microbatches, circular_repeats - 1
        )
        cr_params = jax.tree.map(
            lambda x, cr_idx=cr_idx: jax.lax.dynamic_slice_in_dim(x, cr_idx, 1)[0],
            staged_params)

        # Input for this virtual stage (a repeat inside a stage).
        # Only stage 0 picks up from the first position of the microbatches as input
        # because we always rotate all the staged inputs after one virtual stage.
        # Other stages always pick up activations or garbage values.
        first_stage_input = jnp.where(
            i < num_microbatches,
            jax.lax.dynamic_slice_in_dim(staged_inputs, i, 1)[0],
            states[0]
        )
        states = states.at[0].set(
            jnp.where(stage_index == 0, first_stage_input, states[0])
        )

        # Run stage function for this virtual stage.
        states = states.at[0].set(stage_apply(cr_params, keys[cr_idx], states[0]))

        # Rotate and write output before we rotate states.
        output = jnp.where(
            (stage_index == num_stages - 1) & (i >= output_start_idx),
            states[0],
            outputs[0]
        )
        outputs = jax.lax.dynamic_update_slice_in_dim(
          outputs,
          output[None],
          (i - output_start_idx) % num_microbatches,
          axis=0,
        )
        return states, outputs

      # Pipeline collective permutes for states.
      # The natural way to write the loop is the following:
      #   for i in range(num_iterations):
      #     inps, s, oups = _compute_fn(c)
      #     s = _epilogue_fn(s)
      #
      # To overlap the state permutes with the compute, we can manually pipeline
      # the above loop:
      #   inps, s, oups = _compute_fn(c)     # Iteration K's compute
      #   for i in range(1, num_iterations):
      #     s = _epilogue_fn(s)              # Iteration K's epilogue
      #     inps, s, oups = _compute_fn(c)   # Iteration K+1's compute
      #   ...                                # The last epilogue
      def _body_fn(c: PipelineCarry, _):
        # Parameters for the current virtual stage.
        states, outputs = _compute_fn(
            PipelineCarry(
                states=c.states,
                outputs=c.outputs,
                i=c.i
            )
        )
        with jax.named_scope("ppermute_states"):
          states = _rotate_state_to_right(states, num_stages=num_stages)
        return PipelineCarry(states=states, outputs=outputs, i=c.i + 1), None

      final_carry, _ = jax.lax.scan(
          _body_fn,
          PipelineCarry(
              states=states,
              outputs=outputs,
              i=0
          ),
          length=num_iterations,
      )
      return final_carry.outputs

    return _apply(staged_params, staged_keys, staged_inputs)

  def pipelined_fn(inputs):
    staged_inputs = jax.tree.map(
        partial(_to_microbatches, num_microbatches=num_microbatches), inputs
    )
    if m_sharding is not None:
      staged_inputs = jax.lax.with_sharding_constraint(
          staged_inputs, m_sharding)
    # Reorder keys to follow the circular repeat semantics.
    #  For instance, (0, 1, 2, 3) ==> (0, 2, 1, 3), if we have 2 stages and 2 repeat.
    #    Device 0: 0     2
    #    Device 1:    1     3
    staged_keys = (
        hk.next_rng_keys(num_repeated_stages).reshape(-1,
                                                      num_stages).transpose()
    )
    lifted_init = hk.transparent_lift(pipelined_init_fn)
    # The final shape of each leave of the pytree will be (NUM_STAGE * CIRCULAR_REPEAT, ...)
    # After we shard the params with P('stage'), each stage will receive a circular_repeats
    # number of sets. In the above example, stage 0 will receive [params_0,
    # params_2].
    stage_params = lifted_init(staged_keys, staged_inputs)
    outputs = pipelined_apply_fn(stage_params, staged_keys, staged_inputs)
    return jax.tree.map(lambda x, ref: x.reshape(ref.shape),
                        outputs[-num_microbatches:], inputs)

  return pipelined_fn
