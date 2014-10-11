#include <matrix.h>
#include <mex.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "mattypes.h"
#include "qpgen.h"

typedef double real_t;

// Converts from one type of int array to another.
template <typename T1, typename T2>
void IntToInt(size_t n, const T1 *in, T2 *out) {
  for (size_t i = 0; i < n; ++i)
    out[i] = static_cast<T2>(in[i]);
}

// Initializes dense/sparse/diagonal matrices.
template <typename T>
Dense<T> InitDense(mxArray *arr) {
  Dense<T> mat;
  mat.ord = COL;
  mat.val = reinterpret_cast<T*>(mxGetData(arr));
  mat.m = mxGetM(arr);
  mat.n = mat.lda = mxGetN(arr);
  return mat;
}

template <typename T>
Sparse<T> InitSparse(std::vector<MAT_INT*> *gc, mxArray *arr) {
  mwIndex *mw_row_ind = mxGetIr(arr);
  mwIndex *mw_col_ptr = mxGetJc(arr);

  Sparse<T> mat;
  mat.ord = COL;
  mat.m = mxGetM(arr);
  mat.n = mxGetN(arr);
  mat.nnz = mw_col_ptr[mat.n];

  MAT_INT *row_ind = new MAT_INT[mat.nnz];
  MAT_INT *col_ptr = new MAT_INT[mat.n + 1];

  IntToInt(mat.nnz, mw_row_ind, row_ind);
  IntToInt(mat.n + 1, mw_col_ptr, col_ptr);

  mat.val = reinterpret_cast<T*>(mxGetData(arr));
  mat.ind = row_ind;
  mat.ptr = col_ptr;

  // Add row_ind and col_ptr to garbage collection list.
  gc->push_back(row_ind);
  gc->push_back(col_ptr);

  return mat;
}

template <typename T>
Diagonal<T> InitDiagonal(mxArray *arr) {
  Diagonal<T> mat;
  mat.val = reinterpret_cast<T*>(mxGetData(arr));
  mat.m = mat.n = std::max(mxGetN(arr), mxGetM(arr));
  return mat;
}

// Problem is struct with variables:
// - (dense) F, M, Q1, Q2, R
// - (dense - optional) L1, L2, U1, U2
// - (diagonal) D, Dinv
// - (dense/sparse/diagona) C, L, U
// - (vector) t1, t2
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParam", "No parameters specified");
    return;
  }

  // Get parameter indices.
  const mxArray *params = prhs[0];
  int F_idx = mxGetFieldNumber(params, "F");
  int M_idx = mxGetFieldNumber(params, "M");
  int Q1_idx = mxGetFieldNumber(params, "Q1");
  int Q2_idx = mxGetFieldNumber(params, "Q2");
  int R_idx = mxGetFieldNumber(params, "R");
  int L1_idx = mxGetFieldNumber(params, "L1");
  int L2_idx = mxGetFieldNumber(params, "L2");
  int U1_idx = mxGetFieldNumber(params, "U1");
  int U2_idx = mxGetFieldNumber(params, "U2");
  int D_idx = mxGetFieldNumber(params, "D");
  int Dinv_idx = mxGetFieldNumber(params, "Dinv");
  int C_idx = mxGetFieldNumber(params, "C");
  int L_idx = mxGetFieldNumber(params, "L");
  int U_idx = mxGetFieldNumber(params, "U");
  int t1_idx = mxGetFieldNumber(params, "t1");
  int t2_idx = mxGetFieldNumber(params, "t2");

  // Make sure required parameters are present.
  if (M_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamM", "M must be present");
  if (Q1_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamQ1", "Q1 must be present");
  if (Q2_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamQ2", "Q2 must be present");
  if (D_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamD", "D must be present");
  if (Dinv_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamDinv", "Dinv must be present");
  if (C_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamC", "C must be present");
  if (L_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamL", "L must be present");
  if (U_idx == -1)
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingParamU", "U must be present");
  if (M_idx == -1 || Q1_idx == -1 || Q2_idx == -1 || D_idx == -1 ||
      Dinv_idx == -1 || C_idx == -1 || L_idx == -1 || U_idx == -1)
    return;

  // Declare variables of known type.
  Dense<real_t> F, L1, L2, M, Q1, Q2, R, U1, U2;
  Diagonal<real_t> D, Dinv;
  Vector<real_t> t1, t2;
  
  // Populate matrices of known type.
  mxArray *arr;
  arr = mxGetFieldByNumber(params, 0, M_idx);
  M.val = reinterpret_cast<real_t*>(mxGetData(arr));
  M.m = mxGetM(arr);
  M.lda = M.n = mxGetN(arr);
  arr = mxGetFieldByNumber(params, 0, Q1_idx);
  Q1.val = reinterpret_cast<real_t*>(mxGetData(arr));
  Q1.m = mxGetM(arr);
  Q1.lda = Q1.n = mxGetN(arr);
  arr = mxGetFieldByNumber(params, 0, Q2_idx);
  Q2.val = reinterpret_cast<real_t*>(mxGetData(arr));
  Q2.m = mxGetM(arr);
  Q2.lda = Q2.n = mxGetN(arr);
  arr = mxGetFieldByNumber(params, 0, D_idx);
  D.val = reinterpret_cast<real_t*>(mxGetData(arr));
  D.m = D.n = std::max(mxGetM(arr), mxGetN(arr));
  arr = mxGetFieldByNumber(params, 0, Dinv_idx);
  Dinv.val = reinterpret_cast<real_t*>(mxGetData(arr));
  Dinv.m = Dinv.n = std::max(mxGetM(arr), mxGetN(arr));


  if (L1_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, L1_idx);
    L1.val = reinterpret_cast<real_t*>(mxGetData(arr));
    L1.m = mxGetM(arr);
    L1.lda = L1.n = mxGetN(arr);
  }
  if (L2_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, L2_idx);
    L2.val = reinterpret_cast<real_t*>(mxGetData(arr));
    L2.m = mxGetM(arr);
    L2.lda = L2.n = mxGetN(arr);
  }
  if (U1_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, U1_idx);
    U1.val = reinterpret_cast<real_t*>(mxGetData(arr));
    U1.m = mxGetM(arr);
    U1.lda = U1.n = mxGetN(arr);
  }
  if (L2_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, U2_idx);
    U2.val = reinterpret_cast<real_t*>(mxGetData(arr));
    U2.m = mxGetM(arr);
    U2.lda = U2.n = mxGetN(arr);
  }
  if (F_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, F_idx);
    F.val = reinterpret_cast<real_t*>(mxGetData(arr));
    F.m = mxGetM(arr);
    F.lda = F.n = mxGetN(arr);
  }
  if (R_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, R_idx);
    R.val = reinterpret_cast<real_t*>(mxGetData(arr));
    R.m = mxGetM(arr);
    R.lda = R.n = mxGetN(arr);
  }
  if (t1_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, t1_idx);
    t1.val = reinterpret_cast<real_t*>(mxGetData(arr));
    t1.n = std::max(mxGetM(arr), mxGetN(arr));
    t1.stride = 1;
  }
  if (t2_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, t2_idx);
    t2.val = reinterpret_cast<real_t*>(mxGetData(arr));
    t2.n = std::max(mxGetM(arr), mxGetN(arr));
    t2.stride = 1;
  }

  // Declare matrices of unknown types
  mxArray *arr_C = mxGetFieldByNumber(params, 0, C_idx);
  mxArray *arr_L = mxGetFieldByNumber(params, 0, L_idx);
  mxArray *arr_U = mxGetFieldByNumber(params, 0, U_idx);

  enum MatType { kSparse, kDense, kDiagonal };
  MatType c_t, l_t, u_t;
  if (mxIsSparse(arr_C)) {
    c_t = kSparse;
  } else if (std::min(mxGetM(arr_C), mxGetN(arr_C)) == 1) {
    c_t = kDiagonal;
  } else {
    c_t = kDense;
  }
  if (mxIsSparse(arr_L)) {
    l_t = kSparse;
  } else if (std::min(mxGetM(arr_L), mxGetN(arr_L)) == 1) {
    l_t = kDiagonal;
  } else {
    l_t = kDense;
  }
  if (mxIsSparse(arr_U)) {
    u_t = kSparse;
  } else if (std::min(mxGetM(arr_U), mxGetN(arr_U)) == 1) {
    u_t = kDiagonal;
  } else {
    u_t = kDense;
  }

  // Determine Qpgen type.
  Qpgen_t qpgen_t;
  switch (c_t) {
    case kSparse:
      switch (l_t) {
        case kSparse:
          switch (u_t) {
            case kSparse: qpgen_t = DSpSpSp; break;
            case kDense: qpgen_t = DSpSpDe; break;
            case kDiagonal: qpgen_t = DSpSpDi; break;
          }
          break;
        case kDense:
          switch (u_t) {
            case kSparse: qpgen_t = DSpDeSp; break;
            case kDense: qpgen_t = DSpDeDe; break;
            case kDiagonal: qpgen_t = DSpDeDi; break;
          }
          break;
        case kDiagonal:
          switch (u_t) {
            case kSparse: qpgen_t = DSpDiSp; break;
            case kDense: qpgen_t = DSpDiDe; break;
            case kDiagonal: qpgen_t = DSpDiDi; break;
          }
      }
      break;
    case kDense:
      switch (l_t) {
        case kSparse:
          switch (u_t) {
            case kSparse: qpgen_t = DDeSpSp; break;
            case kDense: qpgen_t = DDeSpDe; break;
            case kDiagonal: qpgen_t = DDeSpDi; break;
          }
          break;
        case kDense:
          switch (u_t) {
            case kSparse: qpgen_t = DDeDeSp; break;
            case kDense: qpgen_t = DDeDeDe; break;
            case kDiagonal: qpgen_t = DDeDeDi; break;
          }
          break;
        case kDiagonal:
          switch (u_t) {
            case kSparse: qpgen_t = DDeDiSp; break;
            case kDense: qpgen_t = DDeDiDe; break;
            case kDiagonal: qpgen_t = DDeDiDi; break;
          }
      }
      break;
    case kDiagonal:
      switch (l_t) {
        case kSparse:
          switch (u_t) {
            case kSparse: qpgen_t = DDiSpSp; break;
            case kDense: qpgen_t = DDiSpDe; break;
            case kDiagonal: qpgen_t = DDiSpDi; break;
          }
          break;
        case kDense:
          switch (u_t) {
            case kSparse: qpgen_t = DDiDeSp; break;
            case kDense: qpgen_t = DDiDeDe; break;
            case kDiagonal: qpgen_t = DDiDeDi; break;
          }
          break;
        case kDiagonal:
          switch (u_t) {
            case kSparse: qpgen_t = DDiDiSp; break;
            case kDense: qpgen_t = DDiDiDe; break;
            case kDiagonal: qpgen_t = DDiDiDi; break;
          }
      }
  }

  // Instantiate qpgen.
  void *qpgen = 0;
  std::vector<MAT_INT*> gc;
  switch (qpgen_t) {
    case DDiDiDi: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Diagonal<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpSpSp: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Sparse<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDeDe: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Dense<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeSpSp: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Sparse<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDeSp: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Dense<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpSpDe: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Sparse<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDeSp: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Dense<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeSpDe: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Sparse<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDeDe: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Dense<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDiDi: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Diagonal<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiDiDe: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Diagonal<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiDeDi: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Dense<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDeDi: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Dense<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDiDe: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Diagonal<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiDeDe: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Dense<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDiDi: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Diagonal<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiDiSp: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Diagonal<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiSpDi: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Sparse<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpSpDi: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Sparse<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDiSp: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Diagonal<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiSpSp: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Sparse<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeDiSp: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Diagonal<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDeSpDi: {
      Dense<real_t> C = InitDense<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Dense<real_t>, Sparse<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDeDi: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Diagonal<real_t> U = InitDiagonal<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Dense<real_t>, Diagonal<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DSpDiDe: {
      Sparse<real_t> C = InitSparse<real_t>(&gc, arr_C);
      Diagonal<real_t> L = InitDiagonal<real_t>(arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Sparse<real_t>, Diagonal<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiDeSp: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Dense<real_t> L = InitDense<real_t>(arr_L);
      Sparse<real_t> U = InitSparse<real_t>(&gc, arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Dense<real_t>, Sparse<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case DDiSpDe: {
      Diagonal<real_t> C = InitDiagonal<real_t>(arr_C);
      Sparse<real_t> L = InitSparse<real_t>(&gc, arr_L);
      Dense<real_t> U = InitDense<real_t>(arr_U);
      qpgen = reinterpret_cast<void*>(
          new Qpgen<real_t, Diagonal<real_t>, Sparse<real_t>, Dense<real_t> >
          (C, D, Dinv, F, L, L1, L2, M, Q1, Q2, R, U, U1, U2, t1, t2));
      break;
    } case SDiDiDi:
      case SSpSpSp:
      case SDeDeDe:
      case SDeSpSp:
      case SSpDeSp:
      case SSpSpDe:
      case SDeDeSp:
      case SDeSpDe:
      case SSpDeDe:
      case SDeDiDi:
      case SDiDiDe:
      case SDiDeDi:
      case SDeDeDi:
      case SDeDiDe:
      case SDiDeDe:
      case SSpDiDi:
      case SdiDiSp:
      case SDiSpDi:
      case SSpSpDi:
      case SSpDiSp:
      case SDiSpSp:
      case SDeDiSp:
      case SDeSpDi:
      case SSpDeDi:
      case SSpDiDe:
      case SDiDeSp:
      case SDiSpDe:
      default:
        mexErrMsgIdAndTxt("MATLAB:qpgen:notImplemented", "Float not supported");
  }

  // Clean up temporary data.
  for (size_t i = 0; i < gc.size(); ++i)
    delete [] gc[i];

  // Set up vector for output.
  Vector<real_t> x;
  x.n = mxGetN(arr_C);
  x.val = new real_t[x.n];
  x.stride = 1;

  // Set output as character array.
  void *qpgen_data = reinterpret_cast<void*>(
      new QpgenData<real_t>{qpgen_t, qpgen, x});
  mwSize size_void = sizeof(void*);
  plhs[0] = mxCreateCharArray(1, &size_void);
  memcpy(mxGetData(plhs[0]), qpgen_data, size_void);
}

