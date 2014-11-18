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

#ifndef CULINALG_H_
#define CULINALG_H_

#include <cublas_v2.h>
#include <cusparse.h>

// Blas Level 1
cublasStatus_t axpy(cublasHandle_t handle, int n, const float *alpha,
                    const float *x, int incx, float *y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t axpy(cublasHandle_t handle, int n, const double *alpha,
                    const double *x, int incx, double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

// Dense Blas Level 2
cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m,
                    int n, const float *alpha, const float *A, int lda,
                    const float *x, int incx, const float *beta, float *y,
                    int incy) {
  return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
      incy);
}

cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m,
                    int n, const double *alpha, const double *A, int lda,
                    const double *x, int incx, const double *beta, double *y,
                    int incy) {
  return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
      incy);
}

// Sparse Blas Level 2
cusparseStatus_t spmv(cusparseHandle_t handle, cusparseOperation_t transA,
                      int m, int n, int nnz, const float *alpha,
                      cusparseMatDescr_t descrA, const float *val,
                      const int *row_ptr, const int *col_ind, const float *x,
                      const float *beta, float *y) {
  return cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, val, row_ptr,
      col_ind, x, beta, y);
}

cusparseStatus_t spmv(cusparseHandle_t handle, cusparseOperation_t transA,
                      int m, int n, int nnz, const double *alpha,
                      cusparseMatDescr_t descrA, const double *val,
                      const int *row_ptr, const int *col_ind, const double *x,
                      const double *beta, double *y) {
  return cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, val, row_ptr,
      col_ind, x, beta, y);
}

#endif  // CULINALG_H_

