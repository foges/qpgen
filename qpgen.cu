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

#include "linalg.h"
#include "qpgen.h"

namespace qpgen {

// Init copies cpu arrays to gpu arrays and initializes variables.
template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Init(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, M_t M, D_t D, Di_t Dinv, C_t C,
     U_t U, L_t L) {
  // Dimension check.

  // Allocate constants.
  q.M = AllocMat(M, CudaMemcpyHostToDevice);
  q.D = AllocMat(D, CudaMemcpyHostToDevice);
  q.Dinv = AllocMat(Dinv, CudaMemcpyHostToDevice);
  q.C = AllocMat(C, CudaMemcpyHostToDevice);
  q.U = AllocMat(U, CudaMemcpyHostToDevice);
  q.L = AllocMat(L, CudaMemcpyHostToDevice);

  // Allocate variables.
  q.x = AllocVec<T>(1);
  q.y = AllocVec<T>(1);
  q.lambda = AllocVec<T>(1);
  q.bt = AllocVec<T>(1);
  q.gt = AllocVec<T>(1);
  q.lt = AllocVec<T>(1);
  q.ut = AllocVec<T>(1);
}

template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Free(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, T *bt, T *gt, T *lt, T *ut) {
  // Free matrices.
  FreeMat(q.M);
  FreeMat(q.D);
  FreeMat(q.Dinv);
  FreeMat(q.C);
  FreeMat(q.U);
  FreeMat(q.L);
  FreeMat(q.L);

  // Free vectors.
  Freevec(q.x);
  Freevec(q.y);
  Freevec(q.lambda);
  Freevec(q.bt);
  Freevec(q.gt);
  Freevec(q.lt);
  Freevec(q.ut);
}


template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Solve(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, T *bt, T *gt, T *lt, T *ut) {

}



}  // namespace qpgen

