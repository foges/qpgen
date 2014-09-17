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

enum MAT_ORDER = { ROW, COL };
typedef int MAT_INT;

template <typename T>
struct Dense {
  T *val;
  MAT_INT m, n, lda;
  MAT_ORDER ord;
};

template <typename T>
struct Sparse {
  T *val;
  MAT_INT *ptr, *ind;
  MAT_INT m, n, nnz;
  MAT_ORDER ord;
};

template <typename T>
struct Diagonal {
  T *val;
  MAT_INT m, n;
};

template <typename T>
struct Vector {
  T *val;
  MAT_INT n, stride;
};

#endif  // MATTYPES_H_

