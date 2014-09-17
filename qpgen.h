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

#ifndef QPGEN_H_
#define QPGEN_H_

namespace qpgen {

template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
struct Qpgen {
  // Constant matrices.
  M_t M;
  D_t D;
  Di_t Dinv;
  C_t C;
  U_t U;
  L_t L;

  // Variables
  Vector<T> *x, *y, *lambda, *bt, *gt, *lt, *ut;
};

// Init copies cpu arrays to gpu arrays and initializes variables.
template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Init(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, M_t M, D_t D, Di_t Dinv, C_t C,
     U_t U, L_t L);

template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Free(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, T *bt, T *gt, T *lt, T *ut);

template <typename M_t, typename D_t, typename Di_t, typename C_t,
          typename U_t, typename L_t, typename T>
Solve(Qpgen<M_t, D_t, Di_t, C_t, U_t, L_t, T> q, T *bt, T *gt, T *lt, T *ut);

}  // namespace qpgen

#endif  // QPGEN_H_

