#include <matrix.h>
#include <mex.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "mattypes.h"
#include "qpgen.h"

typedef double real_t;

namespace {
// Returns value of pr[idx], with appropriate casting from id to T.
// If id is not a numeric type, then it returns nan.
template <typename T>
inline T GetVal(const void *pr, size_t idx, mxClassID id) {
  switch(id) {
    case mxDOUBLE_CLASS:
      return static_cast<T>(reinterpret_cast<const double*>(pr)[idx]);
    case mxSINGLE_CLASS:
      return static_cast<T>(reinterpret_cast<const float*>(pr)[idx]);
    case mxINT8_CLASS:
      return static_cast<T>(reinterpret_cast<const char*>(pr)[idx]);
    case mxUINT8_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned char*>(pr)[idx]);
    case mxINT16_CLASS:
      return static_cast<T>(reinterpret_cast<const short*>(pr)[idx]);
    case mxUINT16_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned short*>(pr)[idx]);
    case mxINT32_CLASS:
      return static_cast<T>(reinterpret_cast<const int*>(pr)[idx]);
    case mxUINT32_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned int*>(pr)[idx]);
    case mxINT64_CLASS:
      return static_cast<T>(reinterpret_cast<const long*>(pr)[idx]);
    case mxUINT64_CLASS:
      return static_cast<T>(reinterpret_cast<const unsigned long*>(pr)[idx]);
    case mxLOGICAL_CLASS:
      return static_cast<T>(reinterpret_cast<const bool*>(pr)[idx]);
    case mxCELL_CLASS:
    case mxCHAR_CLASS:
    case mxFUNCTION_CLASS:
    case mxSTRUCT_CLASS:
    case mxUNKNOWN_CLASS:
    case mxVOID_CLASS:
    default:
      return std::numeric_limits<T>::quiet_NaN();
  }
}

template <typename T>
mxClassID GetClassId();

template <>
mxClassID GetClassId<double>() {
  return mxDOUBLE_CLASS;
}

template <>
mxClassID GetClassId<float>() {
  return mxSINGLE_CLASS;
}

template <typename T, typename C_t, typename L_t,  typename U_t>
void Solve(Qpgen<T, C_t, L_t, U_t> *qpgen, Vector<real_t> bt,
           Vector<real_t> gt, Vector<real_t> lt, Vector<real_t> ut,
           int max_iter, Vector<real_t> *x) {
  qpgen->Solve(bt, gt, lt, ut, max_iter, x);
}
}  // namespace

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 3) {
    mexErrMsgIdAndTxt("MATLAB:qpgen:missingArguments",
        "qpgen_run takes three arguments");
    return;
  }

  QpgenData<real_t> qpgen_data;
  memcpy(&qpgen_data, mxGetData(prhs[0]), sizeof(void*));

  const mxArray *params = prhs[1];
  int bt_idx = mxGetFieldNumber(params, "bt");
  int gt_idx = mxGetFieldNumber(params, "gt");
  int lt_idx = mxGetFieldNumber(params, "lt");
  int ut_idx = mxGetFieldNumber(params, "ut");

  int max_it = GetVal<real_t>(prhs[2], 0, mxGetClassID(prhs[2]));

  Vector<real_t> bt, gt, lt, ut;
  mxArray *arr;
  if (bt_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, bt_idx);
    bt.val = reinterpret_cast<real_t*>(mxGetData(arr));
    bt.n = std::max(mxGetM(arr), mxGetN(arr));
    bt.stride = 1;
  }
  if (gt_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, gt_idx);
    gt.val = reinterpret_cast<real_t*>(mxGetData(arr));
    gt.n = std::max(mxGetM(arr), mxGetN(arr));
    gt.stride = 1;
  }
  if (lt_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, lt_idx);
    lt.val = reinterpret_cast<real_t*>(mxGetData(arr));
    lt.n = std::max(mxGetM(arr), mxGetN(arr));
    lt.stride = 1;
  }
  if (ut_idx >= 0) {
    arr = mxGetFieldByNumber(params, 0, ut_idx);
    ut.val = reinterpret_cast<real_t*>(mxGetData(arr));
    ut.n = std::max(mxGetM(arr), mxGetN(arr));
    ut.stride = 1;
  }

  Vector<real_t> x = qpgen_data.x;
  Qpgen_t qpgen_t = qpgen_data.qpgen_t;
  switch (qpgen_t) {
    case DDiDiDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Diagonal<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpSpSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
           Sparse<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDeDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Dense<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeSpSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Sparse<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDeSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Dense<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpSpDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Sparse<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDeSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Dense<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeSpDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Sparse<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDeDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Dense<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDiDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Diagonal<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiDiDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Diagonal<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiDeDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Dense<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDeDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Dense<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDiDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Diagonal<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiDeDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Dense<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDiDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Diagonal<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiDiSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Diagonal<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiSpDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Sparse<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpSpDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Sparse<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDiSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Diagonal<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiSpSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Sparse<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeDiSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Diagonal<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDeSpDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Dense<real_t>,
          Sparse<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDeDi: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Dense<real_t>, Diagonal<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DSpDiDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Sparse<real_t>,
          Diagonal<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiDeSp: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Dense<real_t>, Sparse<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
      break;
    } case DDiSpDe: {
      Solve(reinterpret_cast<Qpgen<real_t, Diagonal<real_t>,
          Sparse<real_t>, Dense<real_t> >*>(qpgen_data.qpgen), 
          bt, gt, lt, ut, max_it, &x);
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

  plhs[0] = mxCreateNumericMatrix(x.n, 1, GetClassId<real_t>(), mxREAL);
  memcpy(mxGetData(plhs[0]), x.val, x.n * sizeof(real_t));
}

