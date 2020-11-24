# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# JAX is Autograd and XLA

load("@org_tensorflow//tensorflow/core/platform/default:build_config.bzl", "pyx_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_library")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "kernel_pybind11_helpers",
    hdrs = ["kernel_pybind11_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":kernel_helpers",
        "@com_google_absl//absl/base",
        "@pybind11",
    ],
)

cc_library(
    name = "kernel_helpers",
    hdrs = ["kernel_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "gpu_kernel_helpers",
    srcs = ["gpu_kernel_helpers.cc"],
    hdrs = ["gpu_kernel_helpers.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

pyx_library(
    name = "lapack",
    srcs = ["lapack.pyx"],
    py_deps = ["@org_tensorflow//third_party/py/numpy"],
)

py_library(
    name = "jaxlib",
    srcs = [
        "rocm_prng.py",
        "rocsolver.py",
        "init.py",
        "pocketfft.py",
        "setup.py",
        "version.py",
    ],
    deps = [":pocketfft_flatbuffers_py"],
)

py_library(
    name = "gpu_support",
    deps = [
        ":rocblas_kernels",
        ":rocm_prng_kernels",
        ":rocsolver_kernels",
    ],
)

pybind_extension(
    name = "rocblas_kernels",
    srcs = ["rocblas.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rocblas_kernels",
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_pybind11_helpers",
#        "@org_tensorflow//tensorflow/stream_executor/rocm:rocblas_plugin", # TODO
        "@org_tensorflow//tensorflow/stream_executor/rocm:all_runtime",  
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
#        "@local_config_rocm//rocm:rocblas", # TODO
        "@local_config_rocm//rocm:rocm", # TODO
        "@local_config_rocm//rocm:rocm_headers",
        "@pybind11",
    ],
)

pybind_extension(
    name = "rocsolver_kernels",
    srcs = ["rocsolver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rocsolver_kernels",
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_pybind11_helpers",
        "@org_tensorflow//tensorflow/stream_executor/rocm:rocblas_plugin", # TODO
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:rocm_headers",
        "@pybind11",
    ],
)

rocm_library(
    name = "rocm_prng_kernels_lib",
    srcs = ["rocm_prng_kernels.cu.cc"],
    hdrs = ["rocm_prng_kernels.h"],
    deps = [
        ":gpu_kernel_helpers",
        ":kernel_helpers",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

pybind_extension(
    name = "rocm_prng_kernels",
    srcs = ["rocm_prng_kernels.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "rocm_prng_kernels",
    deps = [
        ":rocm_prng_kernels_lib",
        ":kernel_pybind11_helpers",
        "@local_config_rocm//rocm:rocm_headers",
        "@pybind11",
    ],
)

flatbuffer_cc_library(
    name = "pocketfft_flatbuffers_cc",
    srcs = ["pocketfft.fbs"],
)

flatbuffer_py_library(
    name = "pocketfft_flatbuffers_py",
    srcs = ["pocketfft.fbs"],
)

pybind_extension(
    name = "_pocketfft",
    srcs = ["pocketfft.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_pocketfft",
    deps = [
        ":kernel_pybind11_helpers",
        ":pocketfft_flatbuffers_cc",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@pocketfft",
        "@pybind11",
    ],
)
