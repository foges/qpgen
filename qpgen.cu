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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "linalg.h"
#include "qpgen.h"

// Init copies cpu arrays to gpu arrays and initializes variables.
template <typename T, typename C_t, typename L_t, typename U_t>
Qpgen<T, C_t, L_t, U_t>::Qpgen(const C_t& C, const Diagonal<T>& D,
                               const Diagonal<T>& Dinv,  const Dense<T> F,
                               const L_t& L, const Dense<T>& L1,
                               const Dense<T>& L2, const Dense<T>& M,
                               const Dense<T>& Q1, const Dense<T>& Q2, 
                               const Dense<T>& R, const U_t& U,
                               const Dense<T>& U1, const Dense<T>& U2,
                               const Vector<T>& t1, const Vector<T>& t2)
    : C_(AllocMat(C, cudaMemcpyHostToDevice)),
      D_(AllocMat(D, cudaMemcpyHostToDevice)),
      Dinv_(AllocMat(Dinv, cudaMemcpyHostToDevice)),
      F_(AllocMat(F, cudaMemcpyHostToDevice)),
      L_(AllocMat(L, cudaMemcpyHostToDevice)),
      L1_(AllocMat(L1, cudaMemcpyHostToDevice)),
      L2_(AllocMat(L2, cudaMemcpyHostToDevice)),
      M_(AllocMat(M, cudaMemcpyHostToDevice)),
      Q1_(AllocMat(Q1, cudaMemcpyHostToDevice)),
      Q2_(AllocMat(Q2, cudaMemcpyHostToDevice)),
      R_(AllocMat(R, cudaMemcpyHostToDevice)),
      U_(AllocMat(U, cudaMemcpyHostToDevice)),
      U1_(AllocMat(U1, cudaMemcpyHostToDevice)),
      U2_(AllocMat(U2, cudaMemcpyHostToDevice)),
      t1_(AllocVec(t1, cudaMemcpyHostToDevice)),
      t2_(AllocVec(t2, cudaMemcpyHostToDevice)) {
  // Dimension check.
  int n_x = C_.n;
  int n_y = D_.n;
  int n_bt = Q1_.m;
  int n_gt = Q2_.m;
  int n_lt = C_.n;
  int n_ut = C_.n;

  // Allocate variables.
  x_ = AllocVec<T>(n_x);
  y_ = AllocVec<T>(n_y);
  yy_ = AllocVec<T>(n_y);
  l_ = AllocVec<T>(n_y);
  q_ = AllocVec<T>(n_x);
  u_ = AllocVec<T>(n_y);
  lambda_ = AllocVec<T>(n_y);
  bt_ = AllocVec<T>(n_bt);
  gt_ = AllocVec<T>(n_gt);
  lt_ = AllocVec<T>(n_lt);
  ut_ = AllocVec<T>(n_ut);

  // Initialize Handles.
  CudaLinalgHandle *hdl = new CudaLinalgHandle;
  cusparseCreate(&(hdl->sparse));
  cublasCreate(&(hdl->dense));
  cusparseCreateMatDescr(&(hdl->descr));
  cuda_linalg_hdl_ = reinterpret_cast<void*>(hdl);
}

template <typename T, typename C_t, typename L_t,  typename U_t>
void Qpgen<T, C_t, L_t, U_t>::Free() {
  // Free matrices.
  FreeMat(C_);
  FreeMat(D_);
  FreeMat(Dinv_);
  FreeMat(L_);
  FreeMat(L1_);
  FreeMat(L2_);
  FreeMat(M_);
  FreeMat(Q1_);
  FreeMat(Q2_);
  FreeMat(U_);
  FreeMat(U1_);
  FreeMat(U2_);

  // Free vectors.
  FreeVec(x_);
  FreeVec(y_);
  FreeVec(lambda_);
  FreeVec(bt_);
  FreeVec(gt_);
  FreeVec(lt_);
  FreeVec(ut_);
  FreeVec(yy_);

  // Clean up handles.
  CudaLinalgHandle *hdl = reinterpret_cast<CudaLinalgHandle*>(cuda_linalg_hdl_);
  cusparseDestroy(hdl->sparse);
  cublasDestroy(hdl->dense);
  cusparseDestroyMatDescr(hdl->descr);
  delete hdl;
  cuda_linalg_hdl_ = 0;
}

template <typename T, typename C_t, typename L_t,  typename U_t>
void Qpgen<T, C_t, L_t, U_t>::Solve(const Vector<T>& bt, const Vector<T>& gt,
                                    const Vector<T>& lt, const Vector<T>& ut,
                                    unsigned int max_it, Vector<T> *x) {
  // Extract handle.
  CudaLinalgHandle hdl = *reinterpret_cast<CudaLinalgHandle*>(cuda_linalg_hdl_);

  // Set up constants.
  const T kOne = static_cast<T>(1);
  const T kNegOne = static_cast<T>(-1);
  const T kZero = static_cast<T>(0);

  // Copy data to device.
  if (!bt_.IsEmpty())
    cudaMemcpy(bt_.val, bt.val, bt.n * sizeof(T), cudaMemcpyHostToDevice);
  if (!gt_.IsEmpty())
    cudaMemcpy(gt_.val, gt.val, gt.n * sizeof(T), cudaMemcpyHostToDevice);
  if (!lt_.IsEmpty())
    cudaMemcpy(lt_.val, lt.val, lt.n * sizeof(T), cudaMemcpyHostToDevice);
  if (!ut_.IsEmpty())
    cudaMemcpy(ut_.val, ut.val, ut.n * sizeof(T), cudaMemcpyHostToDevice);

  // q = Q1*gt + Q2*bt
  if (!gt_.IsEmpty()) {
    gemv(hdl, NO_TRANS, &kOne, Q1_, gt_, &kZero, q_);
  } else {
    cudaMemcpy(q_.val, Q1_.val, Q1_.m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  if (!bt_.IsEmpty()) {
    gemv(hdl, NO_TRANS, &kOne, Q2_, bt_, &kOne, q_);
  } else {
    const Vector<T> v = { Q2_.val, Q2_.m, 1 };
    axpy(hdl, &kOne, v, q_);
  }
 
  // l = L*lt - L1*gt - L2*bt;
  if (!lt_.IsEmpty()) {
    gemv(hdl, NO_TRANS, &kOne, L_, lt_, &kZero, l_);
  } else {
    cudaMemcpy(l_.val, L_.val, L_.m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  if (!L1_.IsEmpty()) {
    if (!gt_.IsEmpty()) {
      gemv(hdl, NO_TRANS, &kNegOne, L1_, gt_, &kOne, l_);
    } else {
      const Vector<T> v = { L1_.val, L1_.m, 1 };
      axpy(hdl, &kNegOne, v, l_);
    }
  }
  if (!L2_.IsEmpty()) {
    if (!bt_.IsEmpty()) {
      gemv(hdl, NO_TRANS, &kNegOne, L2_, bt_, &kOne, l_);
    } else {
      const Vector<T> v = { L2_.val, L2_.m, 1 };
      axpy(hdl, &kNegOne, v, l_);
    }
  }

  // u = U*ut - U1*gt - U2*bt;
  if (!ut.IsEmpty()) {
    gemv(hdl, NO_TRANS, &kOne, U_, ut_, &kZero, u_);
  } else {
    cudaMemcpy(u_.val, U_.val, U_.m * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  if (!U1_.IsEmpty()) {
    if (!gt_.IsEmpty()) {
      gemv(hdl, NO_TRANS, &kNegOne, U1_, gt_, &kOne, u_);
    } else {
      const Vector<T> v = { U1_.val, U1_.m, 1 };
      axpy(hdl, &kNegOne, v, u_);
    }
  }
  if (!U2_.IsEmpty()) {
    if (!bt_.IsEmpty()) {
      gemv(hdl, NO_TRANS, &kNegOne, U2_, bt_, &kOne, u_);
    } else {
      const Vector<T> v = { U2_.val, U2_.m, 1 };
      axpy(hdl, &kNegOne, v, u_);
    }
  }
  
  // yy = 0, lambda = 0
  cudaMemset(yy_.val, 0, yy_.n);
  cudaMemset(lambda_.val, 0, lambda_.n);

  for (unsigned int i = 0; i < max_it; ++i) {
    // x = -M*(D*y - lambda) + q;
    axpy(hdl, &kNegOne, lambda_, yy_);
    cudaMemcpy(x_.val, q_.val, x_.n * sizeof(T), cudaMemcpyDeviceToDevice);
    gemv(hdl, NO_TRANS, &kNegOne, M_, yy_, &kOne, x_);

    // y = min(max(l, Dinv*(C*x + lambda)), u);
    gemv(hdl, NO_TRANS, &kOne, C_, x_, &kOne, lambda_);
    gemv(hdl, NO_TRANS, &kOne, Dinv_, lambda_, &kZero, y_);
    thrust::transform(thrust::device_pointer_cast(y_.val),
        thrust::device_pointer_cast(y_.val + y_.n),
        thrust::device_pointer_cast(u_.val),
        thrust::device_pointer_cast(y_.val),
        thrust::minimum<T>());
    thrust::transform(thrust::device_pointer_cast(y_.val),
        thrust::device_pointer_cast(y_.val + y_.n),
        thrust::device_pointer_cast(l_.val),
        thrust::device_pointer_cast(y_.val),
        thrust::maximum<T>());

    // lambda = lambda + C*x - D*y;
    gemv(hdl, NO_TRANS, &kOne, D_, y_, &kZero, yy_);
    axpy(hdl, &kNegOne, yy_, lambda_);
  }

  if (!R_.IsEmpty()) {
    cudaMemcpy(q_.val, x_.val, x_.n * sizeof(T), cudaMemcpyDeviceToDevice);
    gemv(hdl, NO_TRANS, &kOne, R_, q_, &kZero, q_);
  }
  if (!t1_.IsEmpty()) {
    axpy(hdl, &kOne, t1_, x_);
  }
  if (!t2_.IsEmpty()) {
    axpy(hdl, &kOne, t2_, x_);
  }
  if (!F_.IsEmpty()) {
    cudaMemcpy(q_.val, x_.val, x_.n * sizeof(T), cudaMemcpyDeviceToDevice);
    gemv(hdl, NO_TRANS, &kOne, F_, q_, &kZero, x_);
  }

  // Copy solution to host.
  cudaMemcpy(x->val, x_.val, x_.n * sizeof(T), cudaMemcpyDeviceToHost);
}

// Template class instantiation.
template class Qpgen<double, Diagonal<double>, Diagonal<double>, Diagonal<double> >;
template class Qpgen<double, Sparse<double>, Sparse<double>, Sparse<double> >;
template class Qpgen<double, Dense<double>, Dense<double>, Dense<double> >;

template class Qpgen<double, Dense<double>, Sparse<double>, Sparse<double> >;
template class Qpgen<double, Sparse<double>, Dense<double>, Sparse<double> >;
template class Qpgen<double, Sparse<double>, Sparse<double>, Dense<double> >;
template class Qpgen<double, Dense<double>, Dense<double>, Sparse<double> >;
template class Qpgen<double, Dense<double>, Sparse<double>, Dense<double> >;
template class Qpgen<double, Sparse<double>, Dense<double>, Dense<double> >;

template class Qpgen<double, Dense<double>, Diagonal<double>, Diagonal<double> >;
template class Qpgen<double, Diagonal<double>, Diagonal<double>, Dense<double> >;
template class Qpgen<double, Diagonal<double>, Dense<double>, Diagonal<double> >;
template class Qpgen<double, Dense<double>, Dense<double>, Diagonal<double> >;
template class Qpgen<double, Dense<double>, Diagonal<double>, Dense<double> >;
template class Qpgen<double, Diagonal<double>, Dense<double>, Dense<double> >;

template class Qpgen<double, Sparse<double>, Diagonal<double>, Diagonal<double> >;
template class Qpgen<double, Diagonal<double>, Diagonal<double>, Sparse<double> >;
template class Qpgen<double, Diagonal<double>, Sparse<double>, Diagonal<double> >;
template class Qpgen<double, Sparse<double>, Sparse<double>, Diagonal<double> >;
template class Qpgen<double, Sparse<double>, Diagonal<double>, Sparse<double> >;
template class Qpgen<double, Diagonal<double>, Sparse<double>, Sparse<double> >;

template class Qpgen<double, Dense<double>, Diagonal<double>, Sparse<double> >;
template class Qpgen<double, Dense<double>, Sparse<double>, Diagonal<double> >;
template class Qpgen<double, Sparse<double>, Dense<double>, Diagonal<double> >;
template class Qpgen<double, Sparse<double>, Diagonal<double>, Dense<double> >;
template class Qpgen<double, Diagonal<double>, Dense<double>, Sparse<double> >;
template class Qpgen<double, Diagonal<double>, Sparse<double>, Dense<double> >;

template class Qpgen<float, Diagonal<float>, Diagonal<float>, Diagonal<float> >;
template class Qpgen<float, Sparse<float>, Sparse<float>, Sparse<float> >;
template class Qpgen<float, Dense<float>, Dense<float>, Dense<float> >;

template class Qpgen<float, Dense<float>, Sparse<float>, Sparse<float> >;
template class Qpgen<float, Sparse<float>, Dense<float>, Sparse<float> >;
template class Qpgen<float, Sparse<float>, Sparse<float>, Dense<float> >;
template class Qpgen<float, Dense<float>, Dense<float>, Sparse<float> >;
template class Qpgen<float, Dense<float>, Sparse<float>, Dense<float> >;
template class Qpgen<float, Sparse<float>, Dense<float>, Dense<float> >;

template class Qpgen<float, Dense<float>, Diagonal<float>, Diagonal<float> >;
template class Qpgen<float, Diagonal<float>, Diagonal<float>, Dense<float> >;
template class Qpgen<float, Diagonal<float>, Dense<float>, Diagonal<float> >;
template class Qpgen<float, Dense<float>, Dense<float>, Diagonal<float> >;
template class Qpgen<float, Dense<float>, Diagonal<float>, Dense<float> >;
template class Qpgen<float, Diagonal<float>, Dense<float>, Dense<float> >;

template class Qpgen<float, Sparse<float>, Diagonal<float>, Diagonal<float> >;
template class Qpgen<float, Diagonal<float>, Diagonal<float>, Sparse<float> >;
template class Qpgen<float, Diagonal<float>, Sparse<float>, Diagonal<float> >;
template class Qpgen<float, Sparse<float>, Sparse<float>, Diagonal<float> >;
template class Qpgen<float, Sparse<float>, Diagonal<float>, Sparse<float> >;
template class Qpgen<float, Diagonal<float>, Sparse<float>, Sparse<float> >;

template class Qpgen<float, Dense<float>, Diagonal<float>, Sparse<float> >;
template class Qpgen<float, Dense<float>, Sparse<float>, Diagonal<float> >;
template class Qpgen<float, Sparse<float>, Dense<float>, Diagonal<float> >;
template class Qpgen<float, Sparse<float>, Diagonal<float>, Dense<float> >;
template class Qpgen<float, Diagonal<float>, Dense<float>, Sparse<float> >;
template class Qpgen<float, Diagonal<float>, Sparse<float>, Dense<float> >;

