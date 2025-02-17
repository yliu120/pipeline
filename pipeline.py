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


### Debugging Utils ###
# Currently, we need to go from partial-auto to full manual for
# printing. This won't be needed until jax.debug.print is fixed.
# The following is an example.
#
# @partial(
#     shard_map,
#     mesh=mesh,
#     in_specs=(P(), P(), P(None, None)),
#     out_specs=P(None, None),
#     check_rep=False
# )
# def _print_output2(stage_index, i, tensor):
#     jax.debug.print("stage_index: {s}, iteration: {i}, debug tensor: {t}",
#                     s=stage_index, i=i, t=tensor)
#     return tensor
### 


def _to_microbatches(input: ArrayLike, num_microbatches: int = 1):
  # The data batch needs to be in shape of (NUM_MICROBATCHES, MICROBATCH_SIZE, NUM_FEATURES)
  global_batch_size = input.shape[0]
  staged_batch = input.reshape(
      num_microbatches, global_batch_size // num_microbatches, *input.shape[1:]
  )
  return staged_batch


def _pipelined_send(x: ArrayLike, perm: list[tuple[int]]):
  """Sends the payload to the next stage."""
  with jax.named_scope("pipelined_send"):
    return jax.lax.psend(x, None, _STAGE_AXIS, perm=perm)


def _pipelined_recv(x: ArrayLike, perm: list[tuple[int]]):
  """Sends the payload to the next stage."""
  with jax.named_scope("pipelined_recv"):
    return jax.lax.precv(x, x, _STAGE_AXIS, perm=perm)


# Args we carried to the next iteration of the layer scan of the pipeline_fn.
PipelineCarry = namedtuple(
    "PipelineCarry", [
        # Intermediate activations
        "states_from_prev",
        "states_to_next",
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
    # Permutation across stages.
    full_perm = [(i, (i + 1) % num_stages) for i in range(num_stages)]

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
      # Represents input/output states per iteration.
      states = jnp.zeros(
          output_shape_per_microbatch.shape,
          dtype=staged_inputs.dtype)
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
        states, outs, i = c.states_from_prev, c.outputs, c.i
        cr_idx = jnp.minimum(
            (i - stage_index) // num_microbatches, circular_repeats - 1
        )
        cr_params = jax.tree.map(
            lambda x, cr_idx=cr_idx: jax.lax.dynamic_slice_in_dim(x, cr_idx, 1)[0],
            staged_params)

        # Input for this virtual stage (a repeat inside a stage).
        # Only stage 0 picks up from the first position of the microbatches as input
        # or activations stores in the outputs from the previous circular repeats.
        first_stage_input = jnp.where(
            i < num_microbatches,
            jax.lax.dynamic_slice_in_dim(staged_inputs, i, 1)[0],
            jax.lax.dynamic_slice_in_dim(outs, i % num_microbatches, 1)[0],
        )
        states = jnp.where(stage_index == 0, first_stage_input, states)
        # Hack! Defines scopes for states_to_next as it doesn't have any use.
        states = states + c.states_to_next * 0.0

        # Run stage function for this virtual stage.
        states = stage_apply(cr_params, keys[cr_idx], states)

        # Optimizes these with a custom pallas kernel.
        # The XLA representation of selective updates are very inefficient.
        out_index = (i - output_start_idx) % num_microbatches
        out = jnp.where(
            (stage_index == num_stages - 1) & (i >= output_start_idx),
            states,
            outs[out_index]
        )
        outs = jax.lax.dynamic_update_slice_in_dim(
            outs, out[None], out_index, axis=0,
        )
        return states, outs

      # Steady phase was implemented with scan layers as every phase has
      # roughly same steps.
      def _body_fn(c: PipelineCarry, _):
        # For stage 0, places the pipelined circular repeat output in outputs.
        # Offsets the index by 1 to the left as the value was received in the
        # last iteration.
        save_index = (c.i - 1 - num_stages) % num_microbatches
        outs = jax.lax.dynamic_update_slice_in_dim(
            c.outputs,
            jnp.where(
                stage_index == 0, c.states_from_prev, c.outputs[save_index]
            )[None],
            save_index,
            axis=0,
        )
        sent = _pipelined_send(c.states_to_next, full_perm)
        # The above code finishes a round of rotating states to the right.

        states_to_next, outs = _compute_fn(
            PipelineCarry(
                states_from_prev=c.states_from_prev,
                # Hack! Don't DCE this send.
                states_to_next=sent,
                outputs=outs,
                i=c.i
            )
        )
        states_from_prev = _pipelined_recv(sent, full_perm)
        return PipelineCarry(
            states_from_prev=states_from_prev,
            states_to_next=states_to_next,
            outputs=outs,
            i=c.i + 1), None

      states_from_prev = states
      states_to_next = states
      # 1. Ramp-up phase.
      for i in range(num_stages):
        # For peeled iterations, only passes activations across
        # first i stages to avoid reading garbage values.
        with jax.named_scope(f"peeled_iter_{i}"):
          sent = _pipelined_send(states_to_next, full_perm[:i])
          states_to_next, outputs = _compute_fn(
              PipelineCarry(
                  states_from_prev=states_from_prev,
                  states_to_next=sent,
                  outputs=outputs,
                  i=i,
              ))
          # The first stage won't be a receiver in all peeled layers.
          states_from_prev = _pipelined_recv(sent,
              full_perm[:(i+1)] if i != num_stages - 1 else full_perm[:i]
          )

      # 2. Steady phase.
      final_carry, _ = jax.lax.scan(
          _body_fn,
          PipelineCarry(
              states_from_prev=states_from_prev,
              states_to_next=states_to_next,
              outputs=outputs,
              i=num_stages,
          ),
          length=(num_iterations - num_stages),
      )

      with jax.named_scope("peeled_last_send"):
        sent = _pipelined_send(final_carry.states_to_next, full_perm[:-1])
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
        hk.next_rng_keys(num_repeated_stages).reshape(
            -1, num_stages).transpose()
    )
    lifted_init = hk.transparent_lift(pipelined_init_fn)
    # The final shape of each leave of the pytree will be (NUM_STAGE * CIRCULAR_REPEAT, ...)
    # After we shard the params with P('stage'), each stage will receive a circular_repeats
    # number of sets. In the above example, stage 0 will receive
    # [params_0, params_2].
    stage_params = lifted_init(staged_keys, staged_inputs)
    outputs = pipelined_apply_fn(stage_params, staged_keys, staged_inputs)
    return jax.tree.map(lambda x, ref: x.reshape(ref.shape),
                        outputs[-num_microbatches:], inputs)

  return pipelined_fn
