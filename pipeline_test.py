import os

# TODO(yunlongl): Finds a way to avoid this formatting-breaking thing.
XLA_FLAGS = [
    "--xla_dump_to=/tmp/hlos",
    "--xla_dump_hlo_pass_re=.*",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_graph_level=0",
    "--xla_disable_hlo_passes=rematerialization,collective-permute-cycle-decomposer",
    "--xla_gpu_use_memcpy_local_p2p=false",
    "--xla_gpu_enable_pipelined_all_gather=true",
    "--xla_gpu_enable_pipelined_reduce_scatter=true",
    "--xla_gpu_enable_while_loop_double_buffering=true",
    "--xla_gpu_multi_streamed_windowed_einsum=false",
    "--xla_gpu_collective_permute_decomposer_threshold=0",
    "--xla_gpu_experimental_enable_pipeline_parallelism_opt=true",
    # "--xla_gpu_enable_pipelined_p2p=true",
]
os.environ["XLA_FLAGS"] = " ".join(XLA_FLAGS)

import unittest
from functools import partial
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from pipeline import gpipe_spmd_pipeline


class Linear(hk.Module):
  def __init__(
      self, output_size: int, mesh: Any | None = None, name: str | None = None
  ):
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.mesh = mesh
    self.b_init = jnp.zeros

  def with_sharding_constraints(
      self,
      weight_sharding: NamedSharding | None = None,
      bias_sharding: NamedSharding | None = None,
      activation_sharding: NamedSharding | None = None,
  ):
    self.w_sharding = weight_sharding
    self.b_sharding = bias_sharding
    self.a_sharding = activation_sharding
    return self

  def __call__(self, inputs: jax.Array):
    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    stddev = 1.0 / np.sqrt(self.input_size)
    w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
    if self.w_sharding is not None:
      w = jax.lax.with_sharding_constraint(w, self.w_sharding)
    out = jnp.dot(inputs, w)
    if self.a_sharding is not None:
      out = jax.lax.with_sharding_constraint(out, self.a_sharding)

    b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
    if self.b_sharding is not None:
      b = jax.lax.with_sharding_constraint(b, self.b_sharding)
    b = jnp.broadcast_to(b, out.shape)
    out = out + b

    if self.a_sharding is not None:
      out = jax.lax.with_sharding_constraint(out, self.a_sharding)
    return out


def multi_stages(stage_fn: Callable, num_stages: int = 1,
                 circular_repeats: int = 1):
  """Simply stacks the stage_fn num_stages * circular_repeat times.

  This function transforms the `stage_fn` into a new function that repeats
  the `stage_fn` for num_stages * circular_repeat times. The repeats
  are run on all devices in a sequential order (not a SPMD pipeline style)
  """
  num_repeated_stages = num_stages * circular_repeats
  stage_init, stage_apply = hk.transform(stage_fn)

  def _init_fn(keys, input):
    return jax.vmap(stage_init, in_axes=(0, None))(keys, input)

  def _apply_fn(params, input):
    def body_fn(carry, params_per_stage):
      new_carry = stage_apply(params_per_stage, None, carry)
      return new_carry, None

    # In Haiku, we always want to apply JAX transformations to pure functions.
    final_x, _ = jax.lax.scan(
        body_fn, input, params, length=num_repeated_stages)
    return final_x

  def multi_stages_fn(input):
    keys = hk.next_rng_keys(num_repeated_stages)
    lifted_init_fn = hk.transparent_lift(_init_fn)
    params = lifted_init_fn(keys, input)
    return _apply_fn(params, input)

  return multi_stages_fn


class SpmdPipelineTest(unittest.TestCase):
  def setUp(self):
    if "cpu" in jax.devices()[0].device_kind:
      self.skipTest("Skip this tests on CPU.")
    jax.config.update("jax_use_shardy_partitioner", True)
    jax.config.update("jax_logging_level", "INFO")

  def assert_params_allclose(self, actual, expected):
    jax.tree.map(
        lambda x,
        y: np.testing.assert_allclose(
            x,
            y),
        actual,
        expected)

  def test_degenerated_case(self):
    global_batch_size = 16
    hidden_dim = 1024
    num_stages = 1
    num_microbatches = 1

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x):
      w_sharding = NamedSharding(mesh, P(None, "data"))
      b_sharding = NamedSharding(mesh, P("data"))
      x = Linear(hidden_dim, mesh=mesh, name="first").with_sharding_constraints(
          w_sharding, b_sharding
      )(x)
      x = Linear(hidden_dim, mesh=mesh, name="second").with_sharding_constraints(
          w_sharding, b_sharding
      )(x)
      return x

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            mesh=mesh,
            microbatch_sharding=P("data", None),
        )
    )
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    # We still wraps the stage_fn with the multi_stages transformation for
    # controlling the random number generators the initializers use.
    one_stage_fn = hk.transform(multi_stages(stage_fn, num_stages=num_stages))
    non_pipelined_params = jax.jit(one_stage_fn.init)(key, inp)
    non_pipelined_re = jax.jit(
        one_stage_fn.apply)(
        non_pipelined_params, key, inp)

    self.assert_params_allclose(pipelined_params, non_pipelined_params)
    np.testing.assert_allclose(pipelined_re, non_pipelined_re)

  def test_spmd_pipeline_with_no_circular_repeat(self):
    global_batch_size = 16
    hidden_dim = 1024
    num_stages = 4
    num_microbatches = 8

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x):
      w_sharding = NamedSharding(mesh, P(None, "data"))
      b_sharding = NamedSharding(mesh, P("data"))
      a_sharding = NamedSharding(mesh, P("data", None))
      x = Linear(hidden_dim, mesh=mesh, name="first").with_sharding_constraints(
          w_sharding, b_sharding, a_sharding
      )(x)
      x = Linear(hidden_dim, mesh=mesh, name="second").with_sharding_constraints(
          w_sharding, b_sharding, a_sharding
      )(x)
      return x

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            mesh=mesh,
            microbatch_sharding=P("data", None),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    four_stages_fn = hk.transform(multi_stages(stage_fn, num_stages=num_stages))
    non_pipelined_params = jax.jit(four_stages_fn.init)(key, inp)
    non_pipelined_re = jax.jit(
        four_stages_fn.apply)(
        non_pipelined_params, key, inp)

    self.assert_params_allclose(pipelined_params, non_pipelined_params)
    # TODO: Debug the small numerical differences
    np.testing.assert_allclose(pipelined_re, non_pipelined_re, atol=2e-3)

  def test_spmd_pipeline_with_circular_repeat(self):
    global_batch_size = 16
    hidden_dim = 1024
    num_stages = 4
    num_microbatches = 8
    circular_repeats = 2

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x):
      w_sharding = NamedSharding(mesh, P(None, "data"))
      b_sharding = NamedSharding(mesh, P("data"))
      a_sharding = NamedSharding(mesh, P("data", None))
      return Linear(hidden_dim, mesh=mesh, name="first").with_sharding_constraints(
          w_sharding, b_sharding, a_sharding
      )(x)

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P("data", None),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    # The parameters will be organized as [0, 4, 1, 5, 2, 6, 3, 7] as the layers are
    # distributed to the stages in a circular manner.
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    eight_stages_fn = hk.transform(
        multi_stages(
            stage_fn,
            num_stages=num_stages,
            circular_repeats=circular_repeats)
    )
    non_pipelined_params = jax.jit(eight_stages_fn.init)(key, inp)
    non_pipelined_re = jax.jit(
        eight_stages_fn.apply)(
        non_pipelined_params, key, inp)

    # [0, 4, 1, 5, 2, 6, 3, 7] -> [0, 1, 2, 3, 4, 5, 6, 7]
    original_order = (
        np.arange(num_stages * circular_repeats)
        .reshape(num_stages, circular_repeats)
        .transpose()
        .reshape(-1)
    )
    self.assert_params_allclose(
        jax.tree.map(lambda x: x[original_order], pipelined_params),
        non_pipelined_params,
    )
    # TODO: Debug the small numerical differences
    np.testing.assert_allclose(pipelined_re, non_pipelined_re, atol=2e-3)

  def test_spmd_pipeline_with_nested_shard_map(self):
    global_batch_size = 16
    hidden_dim = 1024
    num_stages = 4
    num_microbatches = 8
    circular_repeats = 2

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x):
      w_sharding = NamedSharding(mesh, P(None, "data"))
      b_sharding = NamedSharding(mesh, P("data"))
      a_sharding = NamedSharding(mesh, P("data", None))
      x = Linear(hidden_dim, mesh=mesh, name="first").with_sharding_constraints(
          w_sharding, b_sharding, a_sharding
      )(x)

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=P("data", None),
          out_specs=P("data", None),
          check_rep=False,
      )
      def f(x):
        return jax.nn.softmax(x)

      return f(x)

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P("data", None),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    # The parameters will be organized as [0, 4, 1, 5, 2, 6, 3, 7] as the layers are
    # distributed to the stages in a circular manner.
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)
    jax.block_until_ready(pipelined_re)

    # # Reference
    # eight_stages_fn = hk.transform(
    #     multi_stages(
    #         stage_fn,
    #         num_stages=num_stages,
    #         circular_repeats=circular_repeats)
    # )
    # non_pipelined_params = jax.jit(eight_stages_fn.init)(key, inp)
    # non_pipelined_re = jax.jit(
    #     eight_stages_fn.apply)(
    #     non_pipelined_params, key, inp)

    # # [0, 4, 1, 5, 2, 6, 3, 7] -> [0, 1, 2, 3, 4, 5, 6, 7]
    # original_order = (
    #     np.arange(num_stages * circular_repeats)
    #     .reshape(num_stages, circular_repeats)
    #     .transpose()
    #     .reshape(-1)
    # )
    # self.assert_params_allclose(
    #     jax.tree.map(lambda x: x[original_order], pipelined_params),
    #     non_pipelined_params,
    # )
    # TODO: Debug the small numerical differences
    # np.testing.assert_allclose(pipelined_re, non_pipelined_re, atol=2e-3)

  def test_pipeline_performance(self):
    global_batch_size = 32
    hidden_dim = 8192
    num_stages = 4
    num_microbatches = 8
    circular_repeats = 4

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x):
      w_sharding = NamedSharding(mesh, P(None, "data"))
      b_sharding = NamedSharding(mesh, P("data"))
      a_sharding = NamedSharding(mesh, P(None, "data"))
      return Linear(hidden_dim, mesh=mesh, name="first").with_sharding_constraints(
          w_sharding, b_sharding, a_sharding
      )(x)

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P(None, None),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    # The parameters will be organized as [0, 4, 1, 5, 2, 6, 3, 7] as the layers are
    # distributed to the stages in a circular manner.
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_apply_fn_aot = jax.jit(
        pipelined_fn.apply).lower(
        pipelined_params, key, inp).compile()

    # Reference
    eight_stages_fn = hk.transform(
        multi_stages(
            stage_fn,
            num_stages=num_stages,
            circular_repeats=circular_repeats)
    )
    non_pipelined_params = jax.jit(eight_stages_fn.init)(key, inp)
    non_piplined_apply_fn_aot = jax.jit(
        eight_stages_fn.apply).lower(
        non_pipelined_params, key, inp).compile()

    # Warmup NCCL
    pipelined_re = pipelined_apply_fn_aot(pipelined_params, key, inp)
    non_pipelined_re = non_piplined_apply_fn_aot(non_pipelined_params, key, inp)
    jax.block_until_ready((pipelined_re, non_pipelined_re))

    # Profile
    with jax.profiler.trace("/tmp/tensorboard"):
      results = []
      # for _ in range(10):
      #   results.append(pipelined_apply_fn_aot(pipelined_params, key, inp))
      # jax.block_until_ready(results)

      results.clear()
      input2 = inp
      for _ in range(10):
        input2 = non_piplined_apply_fn_aot(non_pipelined_params, key, input2)
      jax.block_until_ready(input2)


if __name__ == "__main__":
  unittest.main()
