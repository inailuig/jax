/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_runtime_api.h"
//#include "rocm/include/rocsolver.h"
//#include "rocm/include/hipblas.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/gpu_kernel_helpers.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {
namespace {

namespace py = pybind11;

void ThrowIfErrorStatus(rocsolverStatus_t status) {
  switch (status) {
    case rocsolver_STATUS_SUCCESS:
      return;
    case rocsolver_STATUS_NOT_INITIALIZED:
      throw std::runtime_error("rocsolver has not been initialized");
    case rocsolver_STATUS_ALLOC_FAILED:
      throw std::runtime_error("rocsolver allocation failed");
    case rocsolver_STATUS_INVALID_VALUE:
      throw std::runtime_error("rocsolver invalid value error");
    case rocsolver_STATUS_ARCH_MISMATCH:
      throw std::runtime_error("rocsolver architecture mismatch error");
    case rocsolver_STATUS_MAPPING_ERROR:
      throw std::runtime_error("rocsolver mapping error");
    case rocsolver_STATUS_EXECUTION_FAILED:
      throw std::runtime_error("rocsolver execution failed");
    case rocsolver_STATUS_INTERNAL_ERROR:
      throw std::runtime_error("rocsolver internal error");
    case rocsolver_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      throw std::invalid_argument("rocsolver matrix type not supported error");
    case rocsolver_STATUS_NOT_SUPPORTED:
      throw std::runtime_error("rocsolver not supported error");
    case rocsolver_STATUS_ZERO_PIVOT:
      throw std::runtime_error("rocsolver zero pivot error");
    case rocsolver_STATUS_INVALID_LICENSE:
      throw std::runtime_error("rocsolver invalid license error");
    default:
      throw std::runtime_error("Unknown rocsolver error");
  }
}

// To avoid creating rocsolver contexts in the middle of execution, we maintain
// a pool of them.
class SolverHandlePool {
 public:
  SolverHandlePool() = default;

  // RAII class representing a rocsolver handle borrowed from the pool. Returns
  // the handle to the pool on destruction.
  class Handle {
   public:
    Handle() = default;
    ~Handle() {
      if (pool_) {
        pool_->Return(handle_);
      }
    }

    Handle(Handle const&) = delete;
    Handle(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
    }
    Handle& operator=(Handle const&) = delete;
    Handle& operator=(Handle&& other) {
      pool_ = other.pool_;
      handle_ = other.handle_;
      other.pool_ = nullptr;
      other.handle_ = nullptr;
      return *this;
    }

    rocsolverDnHandle_t get() { return handle_; }

   private:
    friend class SolverHandlePool;
    Handle(SolverHandlePool* pool, rocsolverDnHandle_t handle)
        : pool_(pool), handle_(handle) {}
    SolverHandlePool* pool_ = nullptr;
    rocsolverDnHandle_t handle_ = nullptr;
  };

  // Borrows a handle from the pool. If 'stream' is non-null, sets the stream
  // associated with the handle.
  static Handle Borrow(hipStream_t stream = nullptr);

 private:
  static SolverHandlePool* Instance();

  void Return(rocsolverDnHandle_t handle);

  absl::Mutex mu_;
  std::vector<rocsolverDnHandle_t> handles_ ABSL_GUARDED_BY(mu_);
};

/*static*/ SolverHandlePool* SolverHandlePool::Instance() {
  static auto* pool = new SolverHandlePool;
  return pool;
}

/*static*/ SolverHandlePool::Handle SolverHandlePool::Borrow(
    hipStream_t stream) {
  SolverHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  rocsolverDnHandle_t handle;
  if (pool->handles_.empty()) {
    ThrowIfErrorStatus(rocsolverDnCreate(&handle));
  } else {
    handle = pool->handles_.back();
    pool->handles_.pop_back();
  }
  if (stream) {
    ThrowIfErrorStatus(rocsolverDnSetStream(handle, stream));
  }
  return Handle(pool, handle);
}

void SolverHandlePool::Return(rocsolverDnHandle_t handle) {
  absl::MutexLock lock(&mu_);
  handles_.push_back(handle);
}

// Set of types known to rocsolver.
enum class Type {
  F32,
  F64,
  C64,
  C128,
};

// Converts a NumPy dtype to a Type.
Type DtypeToType(const py::dtype& np_type) {
  static auto* types = new absl::flat_hash_map<std::pair<char, int>, Type>({
      {{'f', 4}, Type::F32},
      {{'f', 8}, Type::F64},
      {{'c', 8}, Type::C64},
      {{'c', 16}, Type::C128},
  });
  auto it = types->find({np_type.kind(), np_type.itemsize()});
  if (it == types->end()) {
    throw std::invalid_argument(
        absl::StrFormat("Unsupported dtype %s", py::repr(np_type)));
  }
  return it->second;
}

int SizeOfType(Type type) {
  switch (type) {
    case Type::F32:
      return sizeof(float);
    case Type::F64:
      return sizeof(double);
    case Type::C64:
      return sizeof(hipComplex);
    case Type::C128:
      return sizeof(hipDoubleComplex);
  }
}

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  Type type;
  hipblasFillMode_t uplo;
  std::int64_t batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  std::int64_t workspace_size;
  hipblasFillMode_t uplo =
      lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER;
  if (b == 1) {
    switch (type) {
      case Type::F32:
        ThrowIfErrorStatus(rocsolverDnSpotrf_bufferSize(handle.get(), uplo, n,
                                                       /*A=*/nullptr,
                                                       /*lda=*/n, &lwork));
        workspace_size = lwork * sizeof(float);
        break;
      case Type::F64:
        ThrowIfErrorStatus(rocsolverDnDpotrf_bufferSize(handle.get(), uplo, n,
                                                       /*A=*/nullptr,
                                                       /*lda=*/n, &lwork));
        workspace_size = lwork * sizeof(double);
        break;
      case Type::C64:
        ThrowIfErrorStatus(rocsolverDnCpotrf_bufferSize(handle.get(), uplo, n,
                                                       /*A=*/nullptr,
                                                       /*lda=*/n, &lwork));
        workspace_size = lwork * sizeof(hipComplex);
        break;
      case Type::C128:
        ThrowIfErrorStatus(rocsolverDnZpotrf_bufferSize(handle.get(), uplo, n,
                                                       /*A=*/nullptr,
                                                       /*lda=*/n, &lwork));
        workspace_size = lwork * sizeof(hipDoubleComplex);
        break;
    }
  } else {
    // We use the workspace buffer for our own scratch space.
    workspace_size = sizeof(void*) * b;
  }
  return {workspace_size,
          PackDescriptor(PotrfDescriptor{type, uplo, b, n, lwork})};
}

void Potrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const PotrfDescriptor& d =
      *UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0],
                                 SizeOfType(d.type) * d.batch * d.n * d.n,
                                 hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[2]);
  void* workspace = buffers[3];
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        ThrowIfErrorStatus(rocsolverDnSpotrf(handle.get(), d.uplo, d.n, a, d.n,
                                            static_cast<float*>(workspace),
                                            d.lwork, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        ThrowIfErrorStatus(rocsolverDnDpotrf(handle.get(), d.uplo, d.n, a, d.n,
                                            static_cast<double*>(workspace),
                                            d.lwork, info));
        break;
      }
      case Type::C64: {
        hipComplex* a = static_cast<hipComplex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolverDnCpotrf(handle.get(), d.uplo, d.n, a, d.n,
                                            static_cast<hipComplex*>(workspace),
                                            d.lwork, info));
        break;
      }
      case Type::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolverDnZpotrf(
            handle.get(), d.uplo, d.n, a, d.n,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info));
        break;
      }
    }
  } else {
    auto buffer_ptrs_host = MakeBatchPointers(
        stream, buffers[1], workspace, d.batch, SizeOfType(d.type) * d.n * d.n);
    // Make sure that accesses to buffer_ptrs_host complete before we delete it.
    // TODO(phawkins): avoid synchronization here.
    ThrowIfError(hipStreamSynchronize(stream));
    switch (d.type) {
      case Type::F32: {
        ThrowIfErrorStatus(rocsolverDnSpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<float**>(workspace), d.n,

            info, d.batch));
        break;
      }
      case Type::F64: {
        ThrowIfErrorStatus(rocsolverDnDpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<double**>(workspace), d.n,
            info, d.batch));
        break;
      }
      case Type::C64: {
        ThrowIfErrorStatus(rocsolverDnCpotrfBatched(
            handle.get(), d.uplo, d.n, static_cast<hipComplex**>(workspace), d.n,
            info, d.batch));
        break;
      }
      case Type::C128: {
        ThrowIfErrorStatus(rocsolverDnZpotrfBatched(
            handle.get(), d.uplo, d.n,
            static_cast<hipDoubleComplex**>(workspace), d.n, info, d.batch));
        break;
      }
    }
  }
}

// getrf: LU decomposition

struct GetrfDescriptor {
  Type type;
  int batch, m, n;
};

// Returns the workspace size and a descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  switch (type) {
    case Type::F32:
      ThrowIfErrorStatus(rocsolverDnSgetrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::F64:
      ThrowIfErrorStatus(rocsolverDnDgetrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::C64:
      ThrowIfErrorStatus(rocsolverDnCgetrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::C128:
      ThrowIfErrorStatus(rocsolverDnZgetrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
  }
  return {lwork, PackDescriptor(GetrfDescriptor{type, b, m, n})};
}

void Getrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GetrfDescriptor& d =
      *UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnSgetrf(handle.get(), d.m, d.n, a, d.m,
                                            static_cast<float*>(workspace),
                                            ipiv, info));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnDgetrf(handle.get(), d.m, d.n, a, d.m,
                                            static_cast<double*>(workspace),
                                            ipiv, info));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C64: {
      hipComplex* a = static_cast<hipComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnCgetrf(handle.get(), d.m, d.n, a, d.m,
                                            static_cast<hipComplex*>(workspace),
                                            ipiv, info));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnZgetrf(
            handle.get(), d.m, d.n, a, d.m,
            static_cast<hipDoubleComplex*>(workspace), ipiv, info));
        a += d.m * d.n;
        ipiv += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
}

// geqrf: QR decomposition

struct GeqrfDescriptor {
  Type type;
  int batch, m, n, lwork;
};

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b,
                                               int m, int n) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  switch (type) {
    case Type::F32:
      ThrowIfErrorStatus(rocsolverDnSgeqrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::F64:
      ThrowIfErrorStatus(rocsolverDnDgeqrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::C64:
      ThrowIfErrorStatus(rocsolverDnCgeqrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
    case Type::C128:
      ThrowIfErrorStatus(rocsolverDnZgeqrf_bufferSize(handle.get(), m, n,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, &lwork));
      break;
  }
  return {lwork, PackDescriptor(GeqrfDescriptor{type, b, m, n, lwork})};
}

void Geqrf(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GeqrfDescriptor& d =
      *UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* tau = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnSgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                                            static_cast<float*>(workspace),
                                            d.lwork, info));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* tau = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnDgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                                            static_cast<double*>(workspace),
                                            d.lwork, info));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C64: {
      hipComplex* a = static_cast<hipComplex*>(buffers[1]);
      hipComplex* tau = static_cast<hipComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnCgeqrf(handle.get(), d.m, d.n, a, d.m, tau,
                                            static_cast<hipComplex*>(workspace),
                                            d.lwork, info));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
    case Type::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      hipDoubleComplex* tau = static_cast<hipDoubleComplex*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnZgeqrf(
            handle.get(), d.m, d.n, a, d.m, tau,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info));
        a += d.m * d.n;
        tau += std::min(d.m, d.n);
        ++info;
      }
      break;
    }
  }
}

// orgqr/ungqr: apply elementary Householder transformations

struct OrgqrDescriptor {
  Type type;
  int batch, m, n, k, lwork;
};

// Returns the workspace size and a descriptor for a geqrf operation.
std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, int k) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  switch (type) {
    case Type::F32:
      ThrowIfErrorStatus(rocsolverDnSorgqr_bufferSize(handle.get(), m, n, k,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, /*tau=*/nullptr,
                                                     &lwork));
      break;
    case Type::F64:
      ThrowIfErrorStatus(rocsolverDnDorgqr_bufferSize(handle.get(), m, n, k,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, /*tau=*/nullptr,
                                                     &lwork));
      break;
    case Type::C64:
      ThrowIfErrorStatus(rocsolverDnCungqr_bufferSize(handle.get(), m, n, k,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, /*tau=*/nullptr,
                                                     &lwork));
      break;
    case Type::C128:
      ThrowIfErrorStatus(rocsolverDnZungqr_bufferSize(handle.get(), m, n, k,
                                                     /*A=*/nullptr,
                                                     /*lda=*/m, /*tau=*/nullptr,
                                                     &lwork));
      break;
  }
  return {lwork, PackDescriptor(OrgqrDescriptor{type, b, m, n, k, lwork})};
}

void Orgqr(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const OrgqrDescriptor& d =
      *UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  if (buffers[2] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(
        buffers[2], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[3]);
  void* workspace = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[2]);
      float* tau = static_cast<float*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnSorgqr(handle.get(), d.m, d.n, d.k, a, d.m,
                                            tau, static_cast<float*>(workspace),
                                            d.lwork, info));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[2]);
      double* tau = static_cast<double*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolverDnDorgqr(handle.get(), d.m, d.n, d.k, a, d.m, tau,
                             static_cast<double*>(workspace), d.lwork, info));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::C64: {
      hipComplex* a = static_cast<hipComplex*>(buffers[2]);
      hipComplex* tau = static_cast<hipComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnCungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<hipComplex*>(workspace), d.lwork, info));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
    case Type::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[2]);
      hipDoubleComplex* tau = static_cast<hipDoubleComplex*>(buffers[1]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnZungqr(
            handle.get(), d.m, d.n, d.k, a, d.m, tau,
            static_cast<hipDoubleComplex*>(workspace), d.lwork, info));
        a += d.m * d.n;
        tau += d.k;
        ++info;
      }
      break;
    }
  }
}

// Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd

struct SyevdDescriptor {
  Type type;
  hipblasFillMode_t uplo;
  int batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a syevd operation.
std::pair<int, py::bytes> BuildSyevdDescriptor(const py::dtype& dtype,
                                               bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  rocsolverEigMode_t jobz = rocsolver_EIG_MODE_VECTOR;
  hipblasFillMode_t uplo =
      lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER;
  switch (type) {
    case Type::F32:
      ThrowIfErrorStatus(rocsolverDnSsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork));
      break;
    case Type::F64:
      ThrowIfErrorStatus(rocsolverDnDsyevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork));
      break;
    case Type::C64:
      ThrowIfErrorStatus(rocsolverDnCheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork));
      break;
    case Type::C128:
      ThrowIfErrorStatus(rocsolverDnZheevd_bufferSize(
          handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n, /*W=*/nullptr,
          &lwork));
      break;
  }
  return {lwork, PackDescriptor(SyevdDescriptor{type, uplo, b, n, lwork})};
}

void Syevd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const SyevdDescriptor& d =
      *UnpackDescriptor<SyevdDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  ThrowIfError(hipMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
      hipMemcpyDeviceToDevice, stream));
  rocsolverEigMode_t jobz = rocsolver_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnSsyevd(handle.get(), jobz, d.uplo, d.n, a,
                                            d.n, w, static_cast<float*>(work),
                                            d.lwork, info));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnDsyevd(handle.get(), jobz, d.uplo, d.n, a,
                                            d.n, w, static_cast<double*>(work),
                                            d.lwork, info));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::C64: {
      hipComplex* a = static_cast<hipComplex*>(buffers[1]);
      float* w = static_cast<float*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(
            rocsolverDnCheevd(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                             static_cast<hipComplex*>(work), d.lwork, info));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
    case Type::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      double* w = static_cast<double*>(buffers[2]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnZheevd(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<hipDoubleComplex*>(work), d.lwork, info));
        a += d.n * d.n;
        w += d.n;
        ++info;
      }
      break;
    }
  }
}

// Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
// Supports batches of matrices up to size 32.

struct SyevjDescriptor {
  Type type;
  hipblasFillMode_t uplo;
  int batch, n;
  int lwork;
};

// Returns the workspace size and a descriptor for a syevj_batched operation.
std::pair<int, py::bytes> BuildSyevjDescriptor(const py::dtype& dtype,
                                               bool lower, int batch, int n) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  syevjInfo_t params;
  ThrowIfErrorStatus(rocsolverDnCreateSyevjInfo(&params));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { rocsolverDnDestroySyevjInfo(p); });
  rocsolverEigMode_t jobz = rocsolver_EIG_MODE_VECTOR;
  hipblasFillMode_t uplo =
      lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER;
  if (batch == 1) {
    switch (type) {
      case Type::F32:
        ThrowIfErrorStatus(rocsolverDnSsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params));
        break;
      case Type::F64:
        ThrowIfErrorStatus(rocsolverDnDsyevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params));
        break;
      case Type::C64:
        ThrowIfErrorStatus(rocsolverDnCheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params));
        break;
      case Type::C128:
        ThrowIfErrorStatus(rocsolverDnZheevj_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params));
        break;
    }
  } else {
    switch (type) {
      case Type::F32:
        ThrowIfErrorStatus(rocsolverDnSsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch));
        break;
      case Type::F64:
        ThrowIfErrorStatus(rocsolverDnDsyevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch));
        break;
      case Type::C64:
        ThrowIfErrorStatus(rocsolverDnCheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch));
        break;
      case Type::C128:
        ThrowIfErrorStatus(rocsolverDnZheevjBatched_bufferSize(
            handle.get(), jobz, uplo, n, /*A=*/nullptr, /*lda=*/n,
            /*W=*/nullptr, &lwork, params, batch));
        break;
    }
  }
  return {lwork, PackDescriptor(SyevjDescriptor{type, uplo, batch, n, lwork})};
}

void Syevj(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const SyevjDescriptor& d =
      *UnpackDescriptor<SyevjDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(
        buffers[1], buffers[0],
        SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
            static_cast<std::int64_t>(d.n) * static_cast<std::int64_t>(d.n),
        hipMemcpyDeviceToDevice, stream));
  }
  syevjInfo_t params;
  ThrowIfErrorStatus(rocsolverDnCreateSyevjInfo(&params));
  std::unique_ptr<syevjInfo, void (*)(syevjInfo*)> params_cleanup(
      params, [](syevjInfo* p) { rocsolverDnDestroySyevjInfo(p); });

  rocsolverEigMode_t jobz = rocsolver_EIG_MODE_VECTOR;
  int* info = static_cast<int*>(buffers[3]);
  void* work = buffers[4];
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnSsyevj(handle.get(), jobz, d.uplo, d.n, a,
                                            d.n, w, static_cast<float*>(work),
                                            d.lwork, info, params));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnDsyevj(handle.get(), jobz, d.uplo, d.n, a,
                                            d.n, w, static_cast<double*>(work),
                                            d.lwork, info, params));
        break;
      }
      case Type::C64: {
        hipComplex* a = static_cast<hipComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnCheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<hipComplex*>(work), d.lwork, info, params));
        break;
      }
      case Type::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnZheevj(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<hipDoubleComplex*>(work), d.lwork, info, params));
        break;
      }
    }
  } else {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnSsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<float*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnDsyevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<double*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::C64: {
        hipComplex* a = static_cast<hipComplex*>(buffers[1]);
        float* w = static_cast<float*>(buffers[2]);
        ThrowIfErrorStatus(rocsolverDnCheevjBatched(
            handle.get(), jobz, d.uplo, d.n, a, d.n, w,
            static_cast<hipComplex*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        double* w = static_cast<double*>(buffers[2]);
        ThrowIfErrorStatus(
            rocsolverDnZheevjBatched(handle.get(), jobz, d.uplo, d.n, a, d.n, w,
                                    static_cast<hipDoubleComplex*>(work),
                                    d.lwork, info, params, d.batch));
        break;
      }
    }
  }
}

// Singular value decomposition using QR algorithm: gesvd

struct GesvdDescriptor {
  Type type;
  int batch, m, n;
  int lwork;
  signed char jobu, jobvt;
};

// Returns the workspace size and a descriptor for a gesvd operation.
std::pair<int, py::bytes> BuildGesvdDescriptor(const py::dtype& dtype, int b,
                                               int m, int n, bool compute_uv,
                                               bool full_matrices) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  switch (type) {
    case Type::F32:
      ThrowIfErrorStatus(
          rocsolverDnSgesvd_bufferSize(handle.get(), m, n, &lwork));
      break;
    case Type::F64:
      ThrowIfErrorStatus(
          rocsolverDnDgesvd_bufferSize(handle.get(), m, n, &lwork));
      break;
    case Type::C64:
      ThrowIfErrorStatus(
          rocsolverDnCgesvd_bufferSize(handle.get(), m, n, &lwork));
      break;
    case Type::C128:
      ThrowIfErrorStatus(
          rocsolverDnZgesvd_bufferSize(handle.get(), m, n, &lwork));
      break;
  }
  signed char jobu, jobvt;
  if (compute_uv) {
    if (full_matrices) {
      jobu = jobvt = 'A';
    } else {
      jobu = jobvt = 'S';
    }
  } else {
    jobu = jobvt = 'N';
  }
  return {lwork,
          PackDescriptor(GesvdDescriptor{type, b, m, n, lwork, jobu, jobvt})};
}

void Gesvd(hipStream_t stream, void** buffers, const char* opaque,
           size_t opaque_len) {
  const GesvdDescriptor& d =
      *UnpackDescriptor<GesvdDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  ThrowIfError(hipMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      hipMemcpyDeviceToDevice, stream));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  switch (d.type) {
    case Type::F32: {
      float* a = static_cast<float*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      float* u = static_cast<float*>(buffers[3]);
      float* vt = static_cast<float*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnSgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, d.m, s, u, d.m, vt, d.n,
                                            static_cast<float*>(work), d.lwork,
                                            /*rwork=*/nullptr, info));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case Type::F64: {
      double* a = static_cast<double*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      double* u = static_cast<double*>(buffers[3]);
      double* vt = static_cast<double*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnDgesvd(handle.get(), d.jobu, d.jobvt, d.m,
                                            d.n, a, d.m, s, u, d.m, vt, d.n,
                                            static_cast<double*>(work), d.lwork,
                                            /*rwork=*/nullptr, info));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case Type::C64: {
      hipComplex* a = static_cast<hipComplex*>(buffers[1]);
      float* s = static_cast<float*>(buffers[2]);
      hipComplex* u = static_cast<hipComplex*>(buffers[3]);
      hipComplex* vt = static_cast<hipComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnCgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<hipComplex*>(work), d.lwork, /*rwork=*/nullptr, info));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
    case Type::C128: {
      hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
      double* s = static_cast<double*>(buffers[2]);
      hipDoubleComplex* u = static_cast<hipDoubleComplex*>(buffers[3]);
      hipDoubleComplex* vt = static_cast<hipDoubleComplex*>(buffers[4]);
      for (int i = 0; i < d.batch; ++i) {
        ThrowIfErrorStatus(rocsolverDnZgesvd(
            handle.get(), d.jobu, d.jobvt, d.m, d.n, a, d.m, s, u, d.m, vt, d.n,
            static_cast<hipDoubleComplex*>(work), d.lwork,
            /*rwork=*/nullptr, info));
        a += d.m * d.n;
        s += std::min(d.m, d.n);
        u += d.m * d.m;
        vt += d.n * d.n;
        ++info;
      }
      break;
    }
  }
}

// Singular value decomposition using Jacobi algorithm: gesvdj

struct GesvdjDescriptor {
  Type type;
  int batch, m, n;
  int lwork;
  rocsolverEigMode_t jobz;
};

// Returns the workspace size and a descriptor for a gesvdj operation.
std::pair<int, py::bytes> BuildGesvdjDescriptor(const py::dtype& dtype,
                                                int batch, int m, int n,
                                                bool compute_uv) {
  Type type = DtypeToType(dtype);
  auto handle = SolverHandlePool::Borrow();
  int lwork;
  rocsolverEigMode_t jobz =
      compute_uv ? rocsolver_EIG_MODE_VECTOR : rocsolver_EIG_MODE_NOVECTOR;
  gesvdjInfo_t params;
  ThrowIfErrorStatus(rocsolverDnCreateGesvdjInfo(&params));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { rocsolverDnDestroyGesvdjInfo(p); });
  if (batch == 1) {
    switch (type) {
      case Type::F32:
        ThrowIfErrorStatus(rocsolverDnSgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params));
        break;
      case Type::F64:
        ThrowIfErrorStatus(rocsolverDnDgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params));
        break;
      case Type::C64:
        ThrowIfErrorStatus(rocsolverDnCgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params));
        break;
      case Type::C128:
        ThrowIfErrorStatus(rocsolverDnZgesvdj_bufferSize(
            handle.get(), jobz, /*econ=*/0, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params));
        break;
    }
  } else {
    switch (type) {
      case Type::F32:
        ThrowIfErrorStatus(rocsolverDnSgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch));
        break;
      case Type::F64:
        ThrowIfErrorStatus(rocsolverDnDgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch));
        break;
      case Type::C64:
        ThrowIfErrorStatus(rocsolverDnCgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch));
        break;
      case Type::C128:
        ThrowIfErrorStatus(rocsolverDnZgesvdjBatched_bufferSize(
            handle.get(), jobz, m, n,
            /*A=*/nullptr, /*lda=*/m, /*S=*/nullptr,
            /*U=*/nullptr, /*ldu=*/m, /*V=*/nullptr,
            /*ldv=*/n, &lwork, params, batch));
        break;
    }
  }
  return {lwork,
          PackDescriptor(GesvdjDescriptor{type, batch, m, n, lwork, jobz})};
}

void Gesvdj(hipStream_t stream, void** buffers, const char* opaque,
            size_t opaque_len) {
  const GesvdjDescriptor& d =
      *UnpackDescriptor<GesvdjDescriptor>(opaque, opaque_len);
  auto handle = SolverHandlePool::Borrow(stream);
  ThrowIfError(hipMemcpyAsync(
      buffers[1], buffers[0],
      SizeOfType(d.type) * static_cast<std::int64_t>(d.batch) *
          static_cast<std::int64_t>(d.m) * static_cast<std::int64_t>(d.n),
      hipMemcpyDeviceToDevice, stream));
  int* info = static_cast<int*>(buffers[5]);
  void* work = buffers[6];
  gesvdjInfo_t params;
  ThrowIfErrorStatus(rocsolverDnCreateGesvdjInfo(&params));
  std::unique_ptr<gesvdjInfo, void (*)(gesvdjInfo*)> params_cleanup(
      params, [](gesvdjInfo* p) { rocsolverDnDestroyGesvdjInfo(p); });
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnSgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<float*>(work), d.lwork, info, params));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnDgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<double*>(work), d.lwork, info, params));
        break;
      }
      case Type::C64: {
        hipComplex* a = static_cast<hipComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        hipComplex* u = static_cast<hipComplex*>(buffers[3]);
        hipComplex* v = static_cast<hipComplex*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnCgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<hipComplex*>(work), d.lwork, info, params));
        break;
      }
      case Type::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        hipDoubleComplex* u = static_cast<hipDoubleComplex*>(buffers[3]);
        hipDoubleComplex* v = static_cast<hipDoubleComplex*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnZgesvdj(
            handle.get(), d.jobz, /*econ=*/0, d.m, d.n, a, d.m, s, u, d.m, v,
            d.n, static_cast<hipDoubleComplex*>(work), d.lwork, info, params));
        break;
      }
    }
  } else {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        float* u = static_cast<float*>(buffers[3]);
        float* v = static_cast<float*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnSgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<float*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        double* u = static_cast<double*>(buffers[3]);
        double* v = static_cast<double*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnDgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<double*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::C64: {
        hipComplex* a = static_cast<hipComplex*>(buffers[1]);
        float* s = static_cast<float*>(buffers[2]);
        hipComplex* u = static_cast<hipComplex*>(buffers[3]);
        hipComplex* v = static_cast<hipComplex*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnCgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<hipComplex*>(work), d.lwork, info, params, d.batch));
        break;
      }
      case Type::C128: {
        hipDoubleComplex* a = static_cast<hipDoubleComplex*>(buffers[1]);
        double* s = static_cast<double*>(buffers[2]);
        hipDoubleComplex* u = static_cast<hipDoubleComplex*>(buffers[3]);
        hipDoubleComplex* v = static_cast<hipDoubleComplex*>(buffers[4]);
        ThrowIfErrorStatus(rocsolverDnZgesvdjBatched(
            handle.get(), d.jobz, d.m, d.n, a, d.m, s, u, d.m, v, d.n,
            static_cast<hipDoubleComplex*>(work), d.lwork, info, params,
            d.batch));
        break;
      }
    }
  }
}

py::dict Registrations() {
  py::dict dict;
  dict["rocsolver_potrf"] = EncapsulateFunction(Potrf);
  dict["rocsolver_getrf"] = EncapsulateFunction(Getrf);
  dict["rocsolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["rocsolver_orgqr"] = EncapsulateFunction(Orgqr);
  dict["rocsolver_syevd"] = EncapsulateFunction(Syevd);
  dict["rocsolver_syevj"] = EncapsulateFunction(Syevj);
  dict["rocsolver_gesvd"] = EncapsulateFunction(Gesvd);
  dict["rocsolver_gesvdj"] = EncapsulateFunction(Gesvdj);
  return dict;
}

PYBIND11_MODULE(rocsolver_kernels, m) {
  m.def("registrations", &Registrations);
  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
  m.def("build_syevj_descriptor", &BuildSyevjDescriptor);
  m.def("build_gesvd_descriptor", &BuildGesvdDescriptor);
  m.def("build_gesvdj_descriptor", &BuildGesvdjDescriptor);
}

}  // namespace
}  // namespace jax
