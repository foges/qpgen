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

#ifndef MATTYPES_H_
#define MATTYPES_H_

enum MAT_ORDER { ROW, COL };
typedef int MAT_INT;

template <typename T>
struct Matrix {
  T *val;
  MAT_INT m, n;

  Matrix() : m(0), n(0) { }
  Matrix(MAT_INT m, MAT_INT n) : m(m), n(n) { }
  bool IsEmpty() const { return m == 0 || n == 0; }
};

template <typename T>
struct Dense : public Matrix<T> {
  MAT_INT lda;
  MAT_ORDER ord;
};

template <typename T>
struct Sparse : public Matrix<T> {
  MAT_INT *ptr, *ind;
  MAT_INT nnz;
  MAT_ORDER ord;
};

template <typename T>
struct Diagonal : public Matrix <T> { };

template <typename T>
struct Vector {
  T *val;
  MAT_INT n, stride;
  bool IsEmpty() const { return n == 0; }
};

#endif  // MATTYPES_H_

