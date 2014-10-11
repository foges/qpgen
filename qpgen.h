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

#include "mattypes.h"

template <typename T, typename C_t, typename L_t,  typename U_t>
class Qpgen {
  // Constants.
  const Dense<T> F_, L1_, L2_, M_, Q1_, Q2_, R_, U1_, U2_;
  const Diagonal<T> D_, Dinv_;
  const C_t C_;
  const L_t L_;
  const U_t U_;
  const Vector<T> t1_, t2_;

  // Variables.
  Vector<T> x_, y_, lambda_, q_, u_, l_, bt_, gt_, lt_, ut_, yy_;

  // Handles.
  void *cuda_linalg_hdl_;

  // Init copies cpu arrays to gpu arrays and initializes variables.
 public:
  Qpgen(const C_t& C,  const Diagonal<T>& D, const Diagonal<T>& Dinv,
        const Dense<T> F, const L_t& L, const Dense<T>& L1,
        const Dense<T>& L2, const Dense<T>& M, const Dense<T>& Q1,
        const Dense<T>& Q2, const Dense<T>& R, const U_t& U,
        const Dense<T>& U1, const Dense<T>& U2, const Vector<T>& t1,
        const Vector<T>& t2);

  void Free();

  void Solve(const Vector<T>& bt, const Vector<T>& gt, const Vector<T>& lt,
             const Vector<T>& ut, unsigned int max_it, Vector<T> *x);
};

// Supported types (T, C_t, L_t, U_t)
enum Qpgen_t { DDiDiDi, DSpSpSp, DDeDeDe,
               DDeSpSp, DSpDeSp, DSpSpDe, DDeDeSp, DDeSpDe, DSpDeDe,
               DDeDiDi, DDiDiDe, DDiDeDi, DDeDeDi, DDeDiDe, DDiDeDe,
               DSpDiDi, DDiDiSp, DDiSpDi, DSpSpDi, DSpDiSp, DDiSpSp,
               DDeDiSp, DDeSpDi, DSpDeDi, DSpDiDe, DDiDeSp, DDiSpDe,
               SDiDiDi, SSpSpSp, SDeDeDe,
               SDeSpSp, SSpDeSp, SSpSpDe, SDeDeSp, SDeSpDe, SSpDeDe,
               SDeDiDi, SDiDiDe, SDiDeDi, SDeDeDi, SDeDiDe, SDiDeDe,
               SSpDiDi, SdiDiSp, SDiSpDi, SSpSpDi, SSpDiSp, SDiSpSp,
               SDeDiSp, SDeSpDi, SSpDeDi, SSpDiDe, SDiDeSp, SDiSpDe };

template <typename T>
struct QpgenData {
  Qpgen_t qpgen_t;
  void *qpgen;
  Vector<T> x;
};

#endif  // QPGEN_H_

