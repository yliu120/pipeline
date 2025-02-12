import os

# TODO(yunlongl): Finds a way to avoid this formatting-breaking thing.
XLA_FLAGS = [
    "--xla_dump_to=/tmp/hlos",
    "--xla_dump_hlo_pass_re=.*",
    "--xla_gpu_enable_latency_hiding_scheduler=false",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_graph_level=0",
    "--xla_disable_hlo_passes=rematerialization",
    # This flag has been very flaky.
    # "--xla_gpu_use_memcpy_local_p2p=true",
    "--xla_gpu_enable_pipelined_all_gather=true",
    "--xla_gpu_enable_pipelined_reduce_scatter=true",
    "--xla_gpu_enable_while_loop_double_buffering=false",
    "--xla_gpu_multi_streamed_windowed_einsum=false",
]

os.environ["XLA_FLAGS"] = " ".join(XLA_FLAGS)

import unittest
from collections import namedtuple
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

  # Specifies shardings for the linear layer.
  ShardingConfig = namedtuple(
    "ShardingConfig", ['w', 'b', 'a'],
    defaults=[None, None, None]
  )

  def __init__(
      self, output_size: int, mesh: Any | None = None, name: str | None = None
  ):
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.mesh = mesh
    self.b_init = jnp.zeros
    self.shardings = self.ShardingConfig()

  def with_sharding_constraints(self, shardings: ShardingConfig):
    self.shardings = shardings
    return self

  def __call__(self, inputs: jax.Array):
    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    stddev = 1.0 / np.sqrt(self.input_size)
    w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
    if self.shardings.w is not None:
      w = jax.lax.with_sharding_constraint(w, self.shardings.w)
    out = jnp.dot(inputs, w)
    if self.shardings.a is not None:
      out = jax.lax.with_sharding_constraint(out, self.shardings.a)
    b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
    if self.shardings.b is not None:
      b = jax.lax.with_sharding_constraint(b, self.shardings.b)
    b = jnp.broadcast_to(b, out.shape)
    out = out + b
    if self.shardings.a is not None:
      out = jax.lax.with_sharding_constraint(out, self.shardings.a)
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

  def _init_fn(keys, input, **kwargs):
    return jax.vmap(stage_init, in_axes=(0, None))(keys, input, **kwargs)

  def _apply_fn(params, input, **kwargs):
    def body_fn(carry, params_per_stage):
      new_carry = stage_apply(params_per_stage, None, carry, **kwargs)
      return new_carry, None

    # In Haiku, we always want to apply JAX transformations to pure functions.
    final_x, _ = jax.lax.scan(
        body_fn, input, params, length=num_repeated_stages)
    return final_x

  def multi_stages_fn(input, **kwargs):
    keys = hk.next_rng_keys(num_repeated_stages)
    lifted_init_fn = hk.transparent_lift(_init_fn)
    params = lifted_init_fn(keys, input, **kwargs)
    return _apply_fn(params, input, **kwargs)

  return multi_stages_fn


class SpmdPipelineTest(unittest.TestCase):
  def setUp(self):
    jax.config.update("jax_use_shardy_partitioner", True)
    if "cpu" in jax.devices()[0].device_kind:
      self.skipTest("Skip this tests on CPU.")

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

    shardings = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, "data")),
      b=NamedSharding(mesh, P("data")),
    )

    def stage_fn(x):
      x = Linear(
        hidden_dim, mesh=mesh, name="first"
      ).with_sharding_constraints(shardings)(x)
      x = Linear(
        hidden_dim, mesh=mesh, name="second"
      ).with_sharding_constraints(shardings)(x)
      return x

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            stage_fn,
            num_stages,
            num_microbatches,
            mesh=mesh,
            microbatch_sharding=P("data"),
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

    shardings_in_pp_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, "data")),
      b=NamedSharding(mesh, P("data")),
      a=NamedSharding(mesh, P(None, "data")),
    )
    def stage_fn(x, shardings=None):
      x = Linear(
        hidden_dim, mesh=mesh, name="first"
      ).with_sharding_constraints(shardings)(x)
      x = Linear(
        hidden_dim, mesh=mesh, name="second"
      ).with_sharding_constraints(shardings)(x)
      return x

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            partial(stage_fn, shardings=shardings_in_pp_tp),
            num_stages,
            num_microbatches,
            mesh=mesh,
            microbatch_sharding=P("data"),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    shardings_in_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, ("stage", "data"))),
      b=NamedSharding(mesh, P(("stage", "data"))),
      a=NamedSharding(mesh, P(None, ("stage", "data"))),
    )
    four_stages_fn = hk.transform(
        multi_stages(
            partial(stage_fn, shardings=shardings_in_tp),
            num_stages=num_stages
        )
    )
    in_out_sharding = NamedSharding(mesh, P(None, ("stage", "data")))
    np_inp = jax.device_put(inp, in_out_sharding)
    non_pipelined_params = jax.jit(four_stages_fn.init)(key, np_inp)
    non_pipelined_re = jax.jit(
        four_stages_fn.apply)(
        non_pipelined_params, key, np_inp)

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

    shardings_in_pp_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, "data")),
      b=NamedSharding(mesh, P("data")),
      a=NamedSharding(mesh, P(None, "data")),
    )
    def stage_fn(x, shardings=None):
      return Linear(
        hidden_dim, mesh=mesh, name="first"
      ).with_sharding_constraints(shardings)(x)

    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            partial(stage_fn, shardings=shardings_in_pp_tp),
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P("data"),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    # The parameters will be organized as [0, 4, 1, 5, 2, 6, 3, 7] as the layers are
    # distributed to the stages in a circular manner.
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    shardings_in_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, ("stage", "data"))),
      b=NamedSharding(mesh, P(("stage", "data"))),
      a=NamedSharding(mesh, P(None, ("stage", "data"))),
    )
    eight_stages_fn = hk.transform(
        multi_stages(
            partial(stage_fn, shardings=shardings_in_tp),
            num_stages=num_stages,
            circular_repeats=circular_repeats)
    )
    in_out_sharding = NamedSharding(mesh, P(None, ("stage", "data")))
    np_inp = jax.device_put(inp, in_out_sharding)
    non_pipelined_params = jax.jit(eight_stages_fn.init)(key, np_inp)
    non_pipelined_re = jax.jit(
        eight_stages_fn.apply)(
        non_pipelined_params, key, np_inp)

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

    shardings_in_pp_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, "data")),
      b=NamedSharding(mesh, P("data")),
      a=NamedSharding(mesh, P(None, "data")),
    )

    def stage_fn(x, shardings=None):
      x = Linear(
        hidden_dim, mesh=mesh, name="first"
      ).with_sharding_constraints(shardings)(x)

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
            partial(stage_fn, shardings=shardings_in_pp_tp),
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P("data"),
        )
    )

    key = jax.random.key(0)
    inp = jax.random.normal(key, (global_batch_size, hidden_dim))

    # The parameters will be organized as [0, 4, 1, 5, 2, 6, 3, 7] as the layers are
    # distributed to the stages in a circular manner.
    pipelined_params = jax.jit(pipelined_fn.init)(key, inp)
    pipelined_re = jax.jit(pipelined_fn.apply)(pipelined_params, key, inp)

    # Reference
    shardings_in_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, ("stage", "data"))),
      b=NamedSharding(mesh, P(("stage", "data"))),
      a=NamedSharding(mesh, P(None, ("stage", "data"))),
    )
    eight_stages_fn = hk.transform(
        multi_stages(
            partial(stage_fn, shardings=shardings_in_tp),
            num_stages=num_stages,
            circular_repeats=circular_repeats)
    )
    in_out_sharding = NamedSharding(mesh, P(None, ("stage", "data")))
    np_inp = jax.device_put(inp, in_out_sharding)
    non_pipelined_params = jax.jit(eight_stages_fn.init)(key, np_inp)
    non_pipelined_re = jax.jit(
        eight_stages_fn.apply)(
        non_pipelined_params, key, np_inp)

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

  def test_pipeline_performance(self):
    global_batch_size = 32
    hidden_dim = 8192
    num_stages = 4
    num_microbatches = 8
    circular_repeats = 4

    mesh = jax.make_mesh(
        (num_stages, jax.device_count() // num_stages), ("stage", "data")
    )

    def stage_fn(x, shardings=None):
      return Linear(
        hidden_dim, mesh=mesh, name="first"
      ).with_sharding_constraints(shardings)(x)

    shardings_in_pp_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, "data")),
      b=NamedSharding(mesh, P("data")),
      a=NamedSharding(mesh, P(None, "data")),
    )
    pipelined_fn = hk.transform(
        gpipe_spmd_pipeline(
            partial(stage_fn, shardings=shardings_in_pp_tp),
            num_stages,
            num_microbatches,
            circular_repeats=circular_repeats,
            mesh=mesh,
            microbatch_sharding=P(None, "data"),
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
    shardings_in_tp = Linear.ShardingConfig(
      w=NamedSharding(mesh, P(None, ("stage", "data"))),
      b=NamedSharding(mesh, P(("stage", "data"))),
      a=NamedSharding(mesh, P(None, ("stage", "data"))),
    )
    eight_stages_fn = hk.transform(
        multi_stages(
            partial(stage_fn, shardings=shardings_in_tp),
            num_stages=num_stages,
            circular_repeats=circular_repeats)
    )
    in_out_sharding = NamedSharding(mesh, P(None, ("stage", "data")))
    np_inp = jax.device_put(inp, in_out_sharding)
    non_pipelined_params = jax.jit(eight_stages_fn.init)(key, np_inp)
    non_piplined_apply_fn_aot = jax.jit(
        eight_stages_fn.apply,
        out_shardings=in_out_sharding).lower(
        non_pipelined_params, key, np_inp).compile()

    # Warmup NCCL
    pipelined_re = pipelined_apply_fn_aot(pipelined_params, key, inp)
    non_pipelined_re = non_piplined_apply_fn_aot(non_pipelined_params, key, inp)
    jax.block_until_ready((pipelined_re, non_pipelined_re))

    # Profile
    with jax.profiler.trace("/tmp/tensorboard"):
      results = []
      for _ in range(10):
        results.append(pipelined_apply_fn_aot(pipelined_params, key, inp))
      jax.block_until_ready(results)

      results.clear()
      for _ in range(10):
        results.append(
            non_piplined_apply_fn_aot(non_pipelined_params, key, np_inp))
      jax.block_until_ready(results)


if __name__ == "__main__":
  unittest.main()
