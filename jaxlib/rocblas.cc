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
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/rocblas.h"
#include "rocm/include/rocsolver.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hip/hip_runtime_api.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "jaxlib/gpu_kernel_helpers.h"
#include "jaxlib/kernel_pybind11_helpers.h"

namespace jax {
namespace {

namespace py = pybind11;

void ThrowIfErrorStatus(rocblas_status status) {
  switch (status) {
    case  rocblas_status_success:
      return;
// TODO
    default:
      throw std::runtime_error("Unknown rocblas error");
  }
}

// To avoid creating hipblas contexts in the middle of execution, we maintain
// a pool of them.
class BlasHandlePool {
 public:
  BlasHandlePool() = default;

  // RAII class representing a cusolver handle borrowed from the pool. Returns
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

    rocblas_handle get() { return handle_; }

   private:
    friend class BlasHandlePool;
    Handle(BlasHandlePool* pool, rocblas_handle handle)
        : pool_(pool), handle_(handle) {}
    BlasHandlePool* pool_ = nullptr;
    rocblas_handle handle_ = nullptr;
  };

  // Borrows a handle from the pool. If 'stream' is non-null, sets the stream
  // associated with the handle.
  static Handle Borrow(hipStream_t stream = nullptr);

 private:
  static BlasHandlePool* Instance();

  void Return(rocblas_handle handle);

  absl::Mutex mu_;
  std::vector<rocblas_handle> handles_ ABSL_GUARDED_BY(mu_);
};

/*static*/ BlasHandlePool* BlasHandlePool::Instance() {
  static auto* pool = new BlasHandlePool;
  return pool;
}

/*static*/ BlasHandlePool::Handle BlasHandlePool::Borrow(hipStream_t stream) {
  BlasHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  rocblas_handle handle;
  if (pool->handles_.empty()) {
    ThrowIfErrorStatus(rocblas_create_handle(&handle));
  } else {
    handle = pool->handles_.back();
    pool->handles_.pop_back();
  }
  if (stream) {
    ThrowIfErrorStatus(rocblas_set_stream(handle, stream));
  }
  return Handle(pool, handle);
}

void BlasHandlePool::Return(rocblas_handle handle) {
  absl::MutexLock lock(&mu_);
  handles_.push_back(handle);
}

// Set of types known to Cusolver.
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
      return sizeof(rocblas_float_complex);
    case Type::C128:
      return sizeof(rocblas_double_complex);
  }
}


//##########################
// rocblas
//##########################


// Batched triangular solve: trsmbatched

struct TrsmBatchedDescriptor {
  Type type;
  int batch, m, n;
  rocblas_side side;
  rocblas_fill uplo;
  rocblas_operation trans;
  rocblas_diagonal diag;
};

// Returns the descriptor for a TrsmBatched operation.
std::pair<size_t, py::bytes> BuildTrsmBatchedDescriptor(
    const py::dtype& dtype, int batch, int m, int n, bool left_side, bool lower,
    bool trans_a, bool conj_a, bool unit_diagonal) {
  size_t size = batch * sizeof(void*);
  TrsmBatchedDescriptor desc;
  desc.type = DtypeToType(dtype);
  desc.batch = batch;
  desc.m = m;
  desc.n = n;
  desc.side = left_side ? rocblas_side_left : rocblas_side_right;
  desc.uplo = lower ? rocblas_fill_lower : rocblas_fill_upper;
  desc.trans = trans_a ? (conj_a ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose) : rocblas_operation_none;
  desc.diag = unit_diagonal ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;
  return {size, PackDescriptor(desc)};
}

void TrsmBatched(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  const TrsmBatchedDescriptor& d =  *UnpackDescriptor<TrsmBatchedDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[2] != buffers[1]) {
    ThrowIfError(hipMemcpyAsync(buffers[2], buffers[1], SizeOfType(d.type) * d.batch * d.m * d.n, hipMemcpyDeviceToDevice, stream));
  }
  const int lda = d.side == rocblas_side_left ? d.m : d.n;
  const int ldb = d.m;
  auto a_batch_host = MakeBatchPointers(stream, buffers[0], buffers[3], d.batch, SizeOfType(d.type) * lda * lda);
  auto b_batch_host = MakeBatchPointers(stream, buffers[2], buffers[4], d.batch, SizeOfType(d.type) * d.m * d.n);
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  ThrowIfError(hipStreamSynchronize(stream));
  switch (d.type) {
    case Type::F32: {
      // float* a = static_cast<float*>(buffers[0]);
      // float* b = static_cast<float*>(buffers[2]);
      float** a_batch_ptrs = static_cast<float**>(buffers[3]);
      float** b_batch_ptrs = static_cast<float**>(buffers[4]);
      // NOTE(phawkins): if alpha is in GPU memory, hipblas seems to segfault.
      const float alpha = 1.0f;
      ThrowIfErrorStatus(rocblas_strsm_batched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<float**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::F64: {
      // double* a = static_cast<double*>(buffers[0]);
      // double* b = static_cast<double*>(buffers[2]);
      double** a_batch_ptrs = static_cast<double**>(buffers[3]);
      double** b_batch_ptrs = static_cast<double**>(buffers[4]);
      const double alpha = 1.0;
      ThrowIfErrorStatus(rocblas_dtrsm_batched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<double**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::C64: {
      // rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[0]);
      // rocblas_float_complex* b = static_cast<rocblas_float_complex*>(buffers[2]);
      rocblas_float_complex** a_batch_ptrs = static_cast<rocblas_float_complex**>(buffers[3]);
      rocblas_float_complex** b_batch_ptrs = static_cast<rocblas_float_complex**>(buffers[4]);
      const rocblas_float_complex alpha = {1.0f, 0.0f};
      ThrowIfErrorStatus(rocblas_ctrsm_batched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<rocblas_float_complex**>(a_batch_ptrs), lda, b_batch_ptrs, ldb,
          d.batch));
      break;
    }
    case Type::C128: {
      // rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[0]);
      // rocblas_double_complex* b = static_cast<rocblas_double_complex*>(buffers[2]);
      rocblas_double_complex** a_batch_ptrs = static_cast<rocblas_double_complex**>(buffers[3]);
      rocblas_double_complex** b_batch_ptrs = static_cast<rocblas_double_complex**>(buffers[4]);
      const rocblas_double_complex alpha = {1.0d, 0.0d};
      ThrowIfErrorStatus(rocblas_ztrsm_batched(
          handle.get(), d.side, d.uplo, d.trans, d.diag, d.m, d.n, &alpha,
          const_cast<rocblas_double_complex**>(a_batch_ptrs), lda, b_batch_ptrs,
          ldb, d.batch));
      break;
    }
  }
}

// Batched LU decomposition: getrfbatched

struct GetrfBatchedDescriptor {
  Type type;
  int batch, n;
};

// Returns the descriptor for a GetrfBatched operation.
std::pair<size_t, py::bytes> BuildGetrfBatchedDescriptor(const py::dtype& dtype, int b, int n) {
  Type type = DtypeToType(dtype);
  size_t size = b * sizeof(void*);
  return {size, PackDescriptor(GetrfBatchedDescriptor{type, b, n})};
}


void GetrfBatched(hipStream_t stream, void** buffers, const char* opaque,
                  size_t opaque_len) {
  const GetrfBatchedDescriptor& d =  *UnpackDescriptor<GetrfBatchedDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[0] != buffers[1]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0], SizeOfType(d.type) * d.batch * d.n * d.n, hipMemcpyDeviceToDevice, stream));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);
  auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch, SizeOfType(d.type) * d.n * d.n);
  // TODO(phawkins): ideally we would not need to synchronize here, but to
  // avoid it we need a way to keep the host-side buffer alive until the copy
  // completes.
  ThrowIfError(hipStreamSynchronize(stream));
  switch (d.type) {
    case Type::F32: {
      // float* a = static_cast<float*>(buffers[1]);
      float** batch_ptrs = static_cast<float**>(buffers[4]);
      ThrowIfErrorStatus(rocsolver_sgetrf_batched(handle.get(), d.n, d.n, batch_ptrs, d.n, ipiv, d.n, info, d.batch));
      break;
    }
    case Type::F64: {
      // double* a = static_cast<double*>(buffers[1]);
      double** batch_ptrs = static_cast<double**>(buffers[4]);
      ThrowIfErrorStatus(rocsolver_dgetrf_batched(handle.get(), d.n, d.n, batch_ptrs, d.n, ipiv, d.n, info, d.batch));
      break;
    }
    case Type::C64: {
      // rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[1]);
      rocblas_float_complex** batch_ptrs = static_cast<rocblas_float_complex**>(buffers[4]);
      ThrowIfErrorStatus(rocsolver_cgetrf_batched(handle.get(), d.n, d.n, batch_ptrs, d.n, ipiv, d.n, info, d.batch));
      break;
    }
    case Type::C128: {
      // rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[1]);
      rocblas_double_complex** batch_ptrs = static_cast<rocblas_double_complex**>(buffers[4]);
      ThrowIfErrorStatus(rocsolver_zgetrf_batched(handle.get(), d.n, d.n, batch_ptrs, d.n, ipiv, d.n, info, d.batch));
      break;
    }
 }
}



//##########################
// rocsolver
//##########################

// potrf: Cholesky decomposition

struct PotrfDescriptor {
  Type type;
  rocblas_fill uplo;
  std::int64_t batch, n;
};

// Returns the descriptor for a potrf operation.
std::pair<int, py::bytes> BuildPotrfDescriptor(const py::dtype& dtype, bool lower, int b, int n) {
  Type type = DtypeToType(dtype);
  rocblas_fill uplo = lower ? rocblas_fill_lower : rocblas_fill_upper;
  std::int64_t size = b * sizeof(void*);
  return {size, PackDescriptor(PotrfDescriptor{type, uplo, b, n})};
}

void Potrf(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  const PotrfDescriptor& d = *UnpackDescriptor<PotrfDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0], SizeOfType(d.type) * d.batch * d.n * d.n, hipMemcpyDeviceToDevice, stream));
  }

  int* info = static_cast<int*>(buffers[2]);
  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_spotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_dpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_cpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_zpotrf(handle.get(), d.uplo, d.n, a, d.n, info));
        break;
      }
    }
  } else {

    auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch, SizeOfType(d.type) * d.n * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** batch_ptrs = static_cast<float**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_spotrf_batched(handle.get(), d.uplo, d.n, batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::F64: {
        double** batch_ptrs = static_cast<double**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_dpotrf_batched(handle.get(), d.uplo, d.n, batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** batch_ptrs = static_cast<rocblas_float_complex**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_cpotrf_batched(handle.get(), d.uplo, d.n, batch_ptrs, d.n, info, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** batch_ptrs = static_cast<rocblas_double_complex**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_zpotrf_batched(handle.get(), d.uplo, d.n, batch_ptrs, d.n, info, d.batch));
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

// Returns the descriptor for a getrf operation.
std::pair<int, py::bytes> BuildGetrfDescriptor(const py::dtype& dtype, int b, int m, int n) {
  Type type = DtypeToType(dtype);
  std::int64_t size = b * sizeof(void*); // TODO
  return {size, PackDescriptor(GetrfDescriptor{type, b, m, n})};
}

void Getrf(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  const GetrfDescriptor& d = *UnpackDescriptor<GetrfDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync(buffers[1], buffers[0], SizeOfType(d.type) * d.batch * d.m * d.n, hipMemcpyDeviceToDevice, stream));
  }

  int* ipiv = static_cast<int*>(buffers[2]);
  int* info = static_cast<int*>(buffers[3]);

  if (d.batch == 1) {
    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_sgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_dgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::C64: {
        rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_cgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[1]);
        ThrowIfErrorStatus(rocsolver_zgetrf(handle.get(), d.m, d.n, a, d.m, ipiv, info));
        break;
      }
    }
  } else {

    auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch, SizeOfType(d.type) * d.m * d.n);
    // TODO(phawkins): ideally we would not need to synchronize here, but to
    // avoid it we need a way to keep the host-side buffer alive until the copy
    // completes.
    ThrowIfError(hipStreamSynchronize(stream));

    switch (d.type) {
      case Type::F32: {
        float** batch_ptrs = static_cast<float**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_sgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, info, d.batch));
        break;
      }
      case Type::F64: {
        double** batch_ptrs = static_cast<double**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_dgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, info, d.batch));
        break;
      }
      case Type::C64: {
        rocblas_float_complex** batch_ptrs = static_cast<rocblas_float_complex**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_cgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, info, d.batch));
        break;
      }
      case Type::C128: {
        rocblas_double_complex** batch_ptrs = static_cast<rocblas_double_complex**>(buffers[4]);
        ThrowIfErrorStatus(rocsolver_zgetrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, info, d.batch));
        break;
      }
    }
  }
}






// geqrf: QR decomposition

struct GeqrfDescriptor {
  Type type;
  int batch, m, n;
};

std::pair<int, py::bytes> BuildGeqrfDescriptor(const py::dtype& dtype, int b, int m, int n) {
  Type type = DtypeToType(dtype);
  std::int64_t size = b * sizeof(void*); // TODO
  return {size, PackDescriptor(GeqrfDescriptor{type, b, m, n})};
}


void Geqrf(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  const GeqrfDescriptor& d = *UnpackDescriptor<GeqrfDescriptor>(opaque, opaque_len);
  auto handle = BlasHandlePool::Borrow(stream);
  if (buffers[1] != buffers[0]) {
    ThrowIfError(hipMemcpyAsync( buffers[1], buffers[0], SizeOfType(d.type) * d.batch * d.m * d.n, hipMemcpyDeviceToDevice, stream));
  }

    if (d.batch == 1) {
      switch (d.type) {
        case Type::F32: {
          float* a = static_cast<float*>(buffers[1]);
          float* ipiv = static_cast<float*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_sgeqrf(handle.get(), d.m, d.n, a, d.m, ipiv));
          break;
        }
        case Type::F64: {
          double* a = static_cast<double*>(buffers[1]);
          double* ipiv = static_cast<double*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_dgeqrf(handle.get(), d.m, d.n, a, d.m, ipiv));
          break;
        }
        case Type::C64: {
          rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[1]);
          rocblas_float_complex* ipiv = static_cast<rocblas_float_complex*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_cgeqrf(handle.get(), d.m, d.n, a, d.m, ipiv));
          break;
        }
        case Type::C128: {
          rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[1]);
          rocblas_double_complex* ipiv = static_cast<rocblas_double_complex*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_zgeqrf(handle.get(), d.m, d.n, a, d.m, ipiv));
          break;
        }
      }
    } else {

      auto a_ptrs_host = MakeBatchPointers(stream, buffers[1], buffers[4], d.batch, SizeOfType(d.type) * d.m * d.n);
      // TODO(phawkins): ideally we would not need to synchronize here, but to
      // avoid it we need a way to keep the host-side buffer alive until the copy
      // completes.
      ThrowIfError(hipStreamSynchronize(stream));

      switch (d.type) {
        case Type::F32: {
          float** batch_ptrs = static_cast<float**>(buffers[4]);
          float* ipiv = static_cast<float*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_sgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, d.batch));
          break;
        }
        case Type::F64: {
          double** batch_ptrs = static_cast<double**>(buffers[4]);
          double* ipiv = static_cast<double*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_dgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, d.batch));
          break;
        }
        case Type::C64: {
          rocblas_float_complex** batch_ptrs = static_cast<rocblas_float_complex**>(buffers[4]);
          rocblas_float_complex* ipiv = static_cast<rocblas_float_complex*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_cgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, d.batch));
          break;
        }
        case Type::C128: {
          rocblas_double_complex** batch_ptrs = static_cast<rocblas_double_complex**>(buffers[4]);
          rocblas_double_complex* ipiv = static_cast<rocblas_double_complex*>(buffers[2]);
          ThrowIfErrorStatus(rocsolver_zgeqrf_batched(handle.get(), d.m, d.n, batch_ptrs, d.m, ipiv, d.m<d.n?d.m:d.n, d.batch));
          break;
        }
      }
    }
  }


  // orgqr/ungqr: apply elementary Householder transformations
  struct OrgqrDescriptor {
    Type type;
    int batch, m, n, k;
  };


  std::pair<int, py::bytes> BuildOrgqrDescriptor(const py::dtype& dtype, int b, int m, int n, int k) {
    Type type = DtypeToType(dtype);
    std::int64_t size = b * sizeof(void*); // TODO
    return {size, PackDescriptor(OrgqrDescriptor{type, b, m, n, k})};
  }

  void Orgqr(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
    const OrgqrDescriptor& d = *UnpackDescriptor<OrgqrDescriptor>(opaque, opaque_len);
    auto handle = BlasHandlePool::Borrow(stream);
    if (buffers[2] != buffers[0]) {
      ThrowIfError(hipMemcpyAsync(buffers[2], buffers[0], SizeOfType(d.type) * d.batch * d.m * d.n, hipMemcpyDeviceToDevice, stream));
    }

    switch (d.type) {
      case Type::F32: {
        float* a = static_cast<float*>(buffers[1]);
        float* ipiv = static_cast<float*>(buffers[2]);
        for (int i = 0; i < d.batch; ++i) {
          ThrowIfErrorStatus(rocsolver_sorgqr(handle.get(), d.m, d.n, d.k, a+(i*d.m*d.n), d.m, ipiv+(i*d.k)));
        }
        break;
      }
      case Type::F64: {
        double* a = static_cast<double*>(buffers[1]);
        double* ipiv = static_cast<double*>(buffers[2]);
        for (int i = 0; i < d.batch; ++i) {
          ThrowIfErrorStatus(rocsolver_dorgqr(handle.get(), d.m, d.n, d.k, a+(i*d.m*d.n), d.m, ipiv+(i*d.k)));
        }
        break;
      }
      case Type::C64:{
        ThrowIfErrorStatus(rocblas_status_not_implemented);
        break;
      }
      case Type::C128:{
        ThrowIfErrorStatus(rocblas_status_not_implemented);
        break;
      }

// not implemented yet in rocsolver
/*
      case Type::C64: {
        rocblas_float_complex* a = static_cast<rocblas_float_complex*>(buffers[1]);
        rocblas_float_complex* ipiv = static_cast<rocblas_float_complex*>(buffers[2]);
        for (int i = 0; i < d.batch; ++i) {
          ThrowIfErrorStatus(rocsolver_corgqr(handle.get(), d.m, d.n, d.k, a+(i*d.m*d.n), d.m, ipiv+(i*d.k)));
        }
        break;
      }
      case Type::C128: {
        rocblas_double_complex* a = static_cast<rocblas_double_complex*>(buffers[1]);
        rocblas_double_complex* ipiv = static_cast<rocblas_double_complex*>(buffers[2]);
        for (int i = 0; i < d.batch; ++i) {
          ThrowIfErrorStatus(rocsolver_zorgqr(handle.get(), d.m, d.n, d.k, a+(i*d.m*d.n), d.m, ipiv+(i*d.k)));
        }
        break;
*/
    }
  }



    // not implemented yet in rocsolver
    // Symmetric (Hermitian) eigendecomposition, QR algorithm: syevd/heevd
    // Symmetric (Hermitian) eigendecomposition, Jacobi algorithm: syevj/heevj
















py::dict Registrations() {
  py::dict dict;

  dict["rocblas_trsm_batched"] = EncapsulateFunction(TrsmBatched);
  dict["rocblas_getrf_batched"] = EncapsulateFunction(GetrfBatched); // TODO WHY is it in both cublas_kernels and cusolver_kernels???

  dict["rocsolver_potrf"] = EncapsulateFunction(Potrf);
  dict["rocsolver_getrf"] = EncapsulateFunction(Getrf);
  dict["rocsolver_geqrf"] = EncapsulateFunction(Geqrf);
  dict["rocsolver_orgqr"] = EncapsulateFunction(Orgqr);
//  dict["rocsolver_syevd"] = EncapsulateFunction(Syevd);
//  dict["rocsolver_syevj"] = EncapsulateFunction(Syevj);

  return dict;
}

PYBIND11_MODULE(rocblas_kernels, m) {
  m.def("registrations", &Registrations);

  m.def("build_trsm_batched_descriptor", &BuildTrsmBatchedDescriptor);
  m.def("build_getrf_batched_descriptor", &BuildGetrfBatchedDescriptor);

  m.def("build_potrf_descriptor", &BuildPotrfDescriptor);
  m.def("build_getrf_descriptor", &BuildGetrfDescriptor);
  m.def("build_geqrf_descriptor", &BuildGeqrfDescriptor);
  m.def("build_orgqr_descriptor", &BuildOrgqrDescriptor);
//  m.def("build_syevd_descriptor", &BuildSyevdDescriptor);
//  m.def("build_syevj_descriptor", &BuildSyevjDescriptor);
}

}  // namespace
}  // namespace jax
