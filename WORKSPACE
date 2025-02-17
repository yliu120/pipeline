workspace(name="pipeline")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "xla",
    commit = "5b43cc241bf8a9738405e3957e8878e31f1e1869",
    remote = "https://github.com/yliu120/xla.git",
)

git_repository(
    name = "jax-src",
    commit = "78da5df2e023fba605afd1834fd701686e7569d8",
    remote = "https://github.com/yliu120/jax.git",
)

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    requirements = {
        "3.10": "@jax-src//build:requirements_lock_3_10.txt",
        "3.11": "@jax-src//build:requirements_lock_3_11.txt",
        "3.12": "@jax-src//build:requirements_lock_3_12.txt",
        "3.13": "@jax-src//build:requirements_lock_3_13.txt",
        "3.13-ft": "@jax-src//build:requirements_lock_3_13_ft.txt",
    },
    local_wheel_inclusion_list = [
        "jaxlib*",
        "jax_cuda*",
        "jax-cuda*",
    ],
    local_wheel_workspaces = ["@jax-src//jaxlib:jax.bzl"],
    local_wheel_dist_folder = "../dist",
    default_python_version = "system",
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pypi_local",
    requirements_lock = "//:requirements.txt",
)
load("@pypi_local//:requirements.bzl", "install_deps")
install_deps()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("@jax-src//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")
