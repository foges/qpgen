////////////////////////////////////////////////////////////////////////////////
// Copyright 2014 Chris Fougner.                                              //
//                                                                            //
// This program is free software: you can redistribute it and/or modify       //
// it under the terms of the GNU General Public License as published by       //
// the Free Software Foundation, either version 3 of the License, or          //
// (at your option) any later version.                                        //
//                                                                            //
// This program is distributed in the hope that it will be useful,            //
// but WITHOUT ANY WARRANTY; without even the implied warranty of             //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              //
// GNU General Public License for more details.                               //
//                                                                            //
// You should have received a copy of the GNU General Public License          //
// along with this program.  If not, see http://www.gnu.org/licenses/.        //
////////////////////////////////////////////////////////////////////////////////

#ifndef LINALG_H_
#define LINALG_H_

#include <assert.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "cu_linalg.h"
#include "mattypes.h"

struct CudaLinalgHandle {
  cusparseHandle_t sparse;
  cublasHandle_t dense;
  cusparseMatDescr_t descr;
};

enum CudaMatOp { TRANS, NO_TRANS };

// Purely local functions
namespace {
cublasOperation_t OpToCublas(CudaMatOp op) {
   if (op == TRANS)
     return CUBLAS_OP_T;
   else
     return CUBLAS_OP_N;
}

cusparseOperation_t OpToCusparse(CudaMatOp op) {
   if (op == TRANS)
     return CUSPARSE_OPERATION_TRANSPOSE;
   else
     return CUSPARSE_OPERATION_NON_TRANSPOSE;
}

cublasOperation_t OpToInvCublas(CudaMatOp op) {
   if (op == NO_TRANS)
     return CUBLAS_OP_T;
   else
     return CUBLAS_OP_N;
}

cusparseOperation_t OpToInvCusparse(CudaMatOp op) {
   if (op == NO_TRANS)
     return CUSPARSE_OPERATION_TRANSPOSE;
   else
     return CUSPARSE_OPERATION_NON_TRANSPOSE;
}
}  // namespace

// Matrix-vector multiply.
template <typename T, typename M>
int gemv(CudaLinalgHandle hdl, CudaMatOp op, const T *alpha, const M A,
         const Vector<T>& x, const T *beta, Vector<T> y);

template <typename T>
int gemv(CudaLinalgHandle hdl, CudaMatOp op, const T *alpha,
         const Sparse<T>& A, const Vector<T>& x, const T *beta,
         Vector<T> *y) {
  // Input check.
  if (op == NO_TRANS) {
    assert(y->n == A.m);
    assert(x.n == A.n);
  } else {
    assert(y->n == A.n);
    assert(x.n == A.m);
  }

  // Multiply.
  cusparseStatus_t err;
  if (A.ord == ROW) {
    err = spmv(hdl.sparse, OpToCusparse(op), A.m, A.n, A.nnz, alpha, hdl.descr,
        A.val, A.ptr, A.ind, x.val, beta, y->val);
  } else {
    err = spmv(hdl.sparse, OpToInvCusparse(op), A.n, A.m, A.nnz, alpha,
        hdl.descr, A.val, A.ptr, A.ind, x.val, beta, y->val);
  }
  return err != CUSPARSE_STATUS_SUCCESS;
}

template <typename T>
int gemv(CudaLinalgHandle hdl, CudaMatOp op, const T *alpha,
         const Dense<T>& A, const Vector<T>& x, const T *beta,
         Vector<T> *y) {
  // Input check.
  if (op == NO_TRANS) {
    assert(y->n == A.m);
    assert(x.n == A.n);
  } else {
    assert(y->n == A.n);
    assert(x.n == A.m);
  }

  // Multiply.
  cublasStatus_t err;
  if (A.ord == COL) {
    err = gemv(hdl.dense, OpToCublas(op), A.m, A.n, alpha, A.val, A.lda, x.val,
        x.stride, beta, y->val, y->stride);
  } else {
    err = gemv(hdl.dense, OpToInvCublas(op), A.n, A.m, alpha, A.val, A.lda,
        x.val, x.stride, beta, y->val, y->stride);
  }
  return err != CUBLAS_STATUS_SUCCESS;
}


template <typename T>
int gemv(CudaLinalgHandle hdl, CudaMatOp op, const T *alpha,
         const Diagonal<T>& A, const Vector<T>& x, const T *beta,
         Vector<T> *y) {
  // Input check.
  if (op == NO_TRANS) {
    assert(y->n == A.m);
    assert(x.n == A.n);
  } else {
    assert(y->n == A.n);
    assert(x.n == A.m);
  }

  // Multiply.
  thrust::transform(thrust::device_pointer_cast(A.val),
      thrust::device_pointer_cast(A.val + std::min(A.m, A.n)),
      thrust::device_pointer_cast(x.val), thrust::device_pointer_cast(y->val),
      thrust::multiplies<T>());
  if (A.m > A.n && op == NO_TRANS)
    cudaMemset(y->val + A.n, 0, (A.m - A.n) * sizeof(T));
  else if (A.m < A.n && op == TRANS)
    cudaMemset(y->val + A.m, 0, (A.n - A.m) * sizeof(T));
  return 0;
}


// Vector addition.
template <typename T>
int axpy(CudaLinalgHandle hdl, const T *alpha, const Vector<T>& x,
         Vector<T> *y) {
  // Input check.
  assert(x.n == y->n);

  // Add.
  cublasStatus_t err = axpy(hdl.dense, x.n, alpha, x.val, x.stride, y->val,
      y->stride);
  return err != CUBLAS_STATUS_SUCCESS;
}

// Matrix allocation.
template <typename T>
Dense<T> AllocMat(const Dense<T>& mat, enum cudaMemcpyKind kind) {
  Dense<T> mat_d;
  mat_d.m = mat.m;
  mat_d.n = mat.n;
  mat_d.lda = mat.lda;
  mat_d.ord = mat.ord;
  mat_d.val = 0;

  MAT_INT dim;
  if (mat.ord == ROW)
    dim = mat.lda * mat.m;
  else
    dim = mat.lda * mat.n;

  if (!mat_d.IsEmpty()) {
    cudaMalloc(&mat_d.val, dim * sizeof(T));
    cudaMemcpy(mat_d.val, mat.val, dim * sizeof(T), kind);
  }
  return mat_d;
}

template <typename T>
Sparse<T> AllocMat(const Sparse<T> mat, enum cudaMemcpyKind kind) {
  Sparse<T> mat_d;
  mat_d.m = mat.m;
  mat_d.n = mat.n;
  mat_d.nnz = mat.nnz;
  mat_d.ord = mat.ord;
  mat_d.val = 0;

  MAT_INT ptr_dim;
  if (mat.ord == ROW)
    ptr_dim = mat.m + 1;
  else
    ptr_dim = mat.n + 1;

  if (!mat_d.IsEmpty()) {
    cudaMalloc(&mat_d.val, mat.nnz * sizeof(T));
    cudaMalloc(&mat_d.ind, mat.nnz * sizeof(MAT_INT));
    cudaMalloc(&mat_d.ptr, ptr_dim * sizeof(MAT_INT));
    cudaMemcpy(mat_d.val, mat.val, mat.nnz * sizeof(T), kind);
    cudaMemcpy(mat_d.ind, mat.ind, mat.nnz * sizeof(MAT_INT), kind);
    cudaMemcpy(mat_d.ptr, mat.ptr, ptr_dim * sizeof(MAT_INT), kind);
  }
  return mat_d;
}

template <typename T>
Diagonal<T> AllocMat(const Diagonal<T>& mat, enum cudaMemcpyKind kind) {
  Diagonal<T> mat_d;
  mat_d.m = mat.m;
  mat_d.n = mat.n;
  mat_d.val = 0;

  if (!mat_d.IsEmpty()) {
    cudaMalloc(&mat_d.val, std::min(mat.n, mat.m) * sizeof(T));
    cudaMemcpy(mat_d.val, mat.val, std::min(mat.n, mat.m) * sizeof(T), kind);
  }
  return mat_d;
}

// Matrix free.
template <typename T>
void FreeMat(const Dense<T>& mat) {
  cudaFree(mat.val);
}

template <typename T>
void FreeMat(const Sparse<T>& mat) {
  cudaFree(mat.val);
  cudaFree(mat.ind);
  cudaFree(mat.ptr);
}

template <typename T>
void FreeMat(const Diagonal<T>& mat) {
  cudaFree(mat.val);
}

// Vector alloc.
template <typename T>
Vector<T> AllocVec(const Vector<T>& vec, enum cudaMemcpyKind kind) {
  Vector<T> vec_d;
  vec_d.n = vec.n;
  vec_d.stride = vec.stride;

  if (!vec_d.IsEmpty()) {
    cudaMalloc(&vec_d.val, vec.n * vec.stride * sizeof(T));
    cudaMemcpy(vec_d.val, vec.val, vec.n * vec.stride * sizeof(T), kind);
  }
  return vec_d;
}

template <typename T>
Vector<T> AllocVec(int n) {
  Vector<T> vec_d;
  vec_d.n = n;
  vec_d.stride = 1;

  if (!vec_d.IsEmpty())
    cudaMalloc(&vec_d.val, vec_d.n * vec_d.stride * sizeof(T));
  return vec_d;
}

template <typename T>
void FreeVec(Vector<T>& vec) {
  cudaFree(vec.val);
}

#endif  // LINALG_H_

