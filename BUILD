load("@jax-src//jaxlib:jax.bzl", "pytype_strict_library", "py_deps")
load("@pypi_local//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test", "py_binary")

py_library(
    name = "pipeline",
    srcs = ["pipeline.py"],
    deps = [
        "@jax-src//jax:jax",
        "@jax-src//jax/extend",
        requirement("dm-haiku"),
    ]
)

py_test(
    name = "pipeline_test",
    srcs = ["pipeline_test.py"],
    deps = [
        ":pipeline",
        "@jax-src//jax:jax",
        "@jax-src//jax/extend",
        "@jax-src//jaxlib/cuda:cuda_gpu_support",
        "@jax-src//jaxlib:cuda_plugin_extension",
        "@jax-src//jax_plugins",
        "@jax-src//jax_plugins/cuda:cuda_plugin",
        requirement("dm-haiku"),
    ] + py_deps("numpy"),
)
