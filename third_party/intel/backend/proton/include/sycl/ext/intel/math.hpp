//
// Modifications, Copyright (C) 2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//
//==-------------- math.hpp - Intel specific math API ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The main header of Intel specific math API
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

// _iml_half_internal is internal representation for fp16 type used in intel
// math device library. The definition here should align with definition in
// https://github.com/intel/llvm/blob/sycl/libdevice/imf_half.hpp
#if defined(__SPIR__) || defined(__SPIRV__)
using _iml_half_internal = _Float16;
#else
#include <cstdint> // for uint16_t
using _iml_half_internal = uint16_t;
#endif

#include <sycl/bit_cast.hpp>
#include <sycl/builtins.hpp>
#include <sycl/ext/intel/math/imf_fp_conversions.hpp>
#include <sycl/ext/intel/math/imf_half_trivial.hpp>
#include <sycl/ext/intel/math/imf_integer_utils.hpp>
#include <sycl/ext/intel/math/imf_rounding_math.hpp>
#include <sycl/ext/intel/math/imf_simd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/bit_cast.hpp>
#include <sycl/half_type.hpp>
#include <type_traits>


namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {

static_assert(sizeof(sycl::half) == sizeof(_iml_half_internal),
              "sycl::half is not compatible with _iml_half_internal.");

/// --------------------------------------------------------------------------
/// ceil(x) function
/// calculate ceiling of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_ceilf16 (_iml_half_internal x);
float __imf_ceilf (float x);
double __imf_ceil (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> ceil (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_ceilf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> ceil(Tp x) {
    return sycl::half2{ceil(x.s0()), ceil(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> ceil (Tp x) {
    return __imf_ceilf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> ceil (Tp x) {
    return __imf_ceil (x);
}


/// --------------------------------------------------------------------------
/// copysign(x, y) function
/// create value with given magnitude, copying sign of second value
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_copysignf16 (_iml_half_internal x, _iml_half_internal y);
float __imf_copysignf (float x, float y);
double __imf_copysign (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> copysign (Tp x, Tp y) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    _iml_half_internal yi = sycl::bit_cast<_iml_half_internal>(y);
    return sycl::bit_cast<sycl::half>(__imf_copysignf16 (xi, yi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> copysign (Tp x, Tp y) {
    return __imf_copysignf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> copysign (Tp x, Tp y) {
    return __imf_copysign (x, y);
}

/// --------------------------------------------------------------------------
/// floor(x) function
/// calculate the largest integer less than or equal to x
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_floorf16 (_iml_half_internal x);
float __imf_floorf (float x);
double __imf_floor (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> floor (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_floorf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> floor (Tp x) {
    return __imf_floorf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> floor (Tp x) {
    return __imf_floor (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> floor(Tp x) {
    return sycl::half2{floor(x.s0()), floor(x.s1())};
}

/// --------------------------------------------------------------------------
/// inv(x) function
/// calculate the inverse of the input argument 1/x
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_invf16 (_iml_half_internal x);
float __imf_invf (float x);
double __imf_inv (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> inv (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_invf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> inv (Tp x) {
    return __imf_invf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> inv (Tp x) {
    return __imf_inv (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> inv(Tp x) {
    return sycl::half2{inv(x.s0()), inv(x.s1())};
}


/// --------------------------------------------------------------------------
/// rint(x) function
/// round input to nearest integer value in floating-point
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_rintf16 (_iml_half_internal x);
float __imf_rintf (float x);
double __imf_rint (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> rint (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_rintf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rint (Tp x) {
    return __imf_rintf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rint (Tp x) {
    return __imf_rint (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> rint(Tp x) {
    return sycl::half2{rint(x.s0()), rint(x.s1())};
}

/// --------------------------------------------------------------------------
/// rsqrt(x) function
/// calculate the reciprocal of the square root of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_rsqrtf16 (_iml_half_internal x);
float __imf_rsqrtf (float x);
double __imf_rsqrt (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> rsqrt (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_rsqrtf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rsqrt (Tp x) {
    return __imf_rsqrtf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rsqrt (Tp x) {
    return __imf_rsqrt (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> rsqrt(Tp x) {
    return sycl::half2{rsqrt(x.s0()), rsqrt(x.s1())};
}

/// --------------------------------------------------------------------------
/// saturate(x) function
/// saturate floating point value
/// --------------------------------------------------------------------------
extern "C" {
float __imf_saturatef (float x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> saturate (Tp x) {
    return __imf_saturatef (x);
}

/// --------------------------------------------------------------------------
/// sqrt(x) function
/// calculate the square root of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_sqrtf16 (_iml_half_internal x);
float __imf_sqrtf (float x);
double __imf_sqrt (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> sqrt (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_sqrtf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sqrt (Tp x) {
    return __imf_sqrtf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sqrt (Tp x) {
    return __imf_sqrt (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> sqrt(Tp x) {
    return sycl::half2{sqrt(x.s0()), sqrt(x.s1())};
}

/// --------------------------------------------------------------------------
/// trunc(x) function
/// truncate input argument to the integral part
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_truncf16 (_iml_half_internal x);
float __imf_truncf (float x);
double __imf_trunc (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> trunc (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_truncf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> trunc (Tp x) {
    return __imf_truncf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> trunc (Tp x) {
    return __imf_trunc (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> trunc(Tp x) {
    return sycl::half2{trunc(x.s0()), trunc(x.s1())};
}

/// --------------------------------------------------------------------------
/// acos(x) function
/// calculate the arc cosine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_acosf (float x);
float __imf_acosf_ha (float x);
float __imf_acosf_la (float x);
float __imf_acosf_ep (float x);
double __imf_acos (double x);
double __imf_acos_ha (double x);
double __imf_acos_la (double x);
double __imf_acos_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acos (Tp x) {
    return __imf_acosf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acos (Tp x) {
    return __imf_acos (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acos (Tp x) {
        return __imf_acosf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acos (Tp x) {
        return __imf_acos_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acos (Tp x) {
        return __imf_acosf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acos (Tp x) {
        return __imf_acos_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acos (Tp x) {
        return __imf_acosf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acos (Tp x) {
        return __imf_acos_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// acosh(x) function
/// calculate the nonnegative inverse hyperbolic cosine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_acoshf (float x);
float __imf_acoshf_ha (float x);
float __imf_acoshf_la (float x);
float __imf_acoshf_ep (float x);
double __imf_acosh (double x);
double __imf_acosh_ha (double x);
double __imf_acosh_la (double x);
double __imf_acosh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acosh (Tp x) {
    return __imf_acoshf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acosh (Tp x) {
    return __imf_acosh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acosh (Tp x) {
        return __imf_acoshf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acosh (Tp x) {
        return __imf_acosh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acosh (Tp x) {
        return __imf_acoshf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acosh (Tp x) {
        return __imf_acosh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> acosh (Tp x) {
        return __imf_acoshf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> acosh (Tp x) {
        return __imf_acosh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// asin(x) function
/// calculate the arc sine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_asinf (float x);
float __imf_asinf_ha (float x);
float __imf_asinf_la (float x);
float __imf_asinf_ep (float x);
double __imf_asin (double x);
double __imf_asin_ha (double x);
double __imf_asin_la (double x);
double __imf_asin_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asin (Tp x) {
    return __imf_asinf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asin (Tp x) {
    return __imf_asin (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asin (Tp x) {
        return __imf_asinf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asin (Tp x) {
        return __imf_asin_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asin (Tp x) {
        return __imf_asinf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asin (Tp x) {
        return __imf_asin_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asin (Tp x) {
        return __imf_asinf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asin (Tp x) {
        return __imf_asin_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// asinh(x) function
/// calculate the inverse hyperbolic sine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_asinhf (float x);
float __imf_asinhf_ha (float x);
float __imf_asinhf_la (float x);
float __imf_asinhf_ep (float x);
double __imf_asinh (double x);
double __imf_asinh_ha (double x);
double __imf_asinh_la (double x);
double __imf_asinh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asinh (Tp x) {
    return __imf_asinhf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asinh (Tp x) {
    return __imf_asinh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asinh (Tp x) {
        return __imf_asinhf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asinh (Tp x) {
        return __imf_asinh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asinh (Tp x) {
        return __imf_asinhf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asinh (Tp x) {
        return __imf_asinh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> asinh (Tp x) {
        return __imf_asinhf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> asinh (Tp x) {
        return __imf_asinh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// atan(x) function
/// calculate the arc tangent of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_atanf (float x);
float __imf_atanf_ha (float x);
float __imf_atanf_la (float x);
float __imf_atanf_ep (float x);
double __imf_atan (double x);
double __imf_atan_ha (double x);
double __imf_atan_la (double x);
double __imf_atan_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan (Tp x) {
    return __imf_atanf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan (Tp x) {
    return __imf_atan (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan (Tp x) {
        return __imf_atanf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan (Tp x) {
        return __imf_atan_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan (Tp x) {
        return __imf_atanf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan (Tp x) {
        return __imf_atan_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan (Tp x) {
        return __imf_atanf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan (Tp x) {
        return __imf_atan_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// atan2(x, y) function
/// calculate the arc tangent of the x/y input arguments
/// --------------------------------------------------------------------------
extern "C" {
float __imf_atan2f (float x, float y);
float __imf_atan2f_ha (float x, float y);
float __imf_atan2f_la (float x, float y);
float __imf_atan2f_ep (float x, float y);
double __imf_atan2 (double x, double y);
double __imf_atan2_ha (double x, double y);
double __imf_atan2_la (double x, double y);
double __imf_atan2_ep (double x, double y);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan2 (Tp x, Tp y) {
    return __imf_atan2f (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan2 (Tp x, Tp y) {
    return __imf_atan2 (x, y);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan2 (Tp x, Tp y) {
        return __imf_atan2f_ha (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan2 (Tp x, Tp y) {
        return __imf_atan2_ha (x, y);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan2 (Tp x, Tp y) {
        return __imf_atan2f_la (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan2 (Tp x, Tp y) {
        return __imf_atan2_la (x, y);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atan2 (Tp x, Tp y) {
        return __imf_atan2f_ep (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atan2 (Tp x, Tp y) {
        return __imf_atan2_ep (x, y);
    }
}

/// --------------------------------------------------------------------------
/// atanh(x) function
/// calculate the inverse hyperbolic tangent of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_atanhf (float x);
float __imf_atanhf_ha (float x);
float __imf_atanhf_la (float x);
float __imf_atanhf_ep (float x);
double __imf_atanh (double x);
double __imf_atanh_ha (double x);
double __imf_atanh_la (double x);
double __imf_atanh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atanh (Tp x) {
    return __imf_atanhf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atanh (Tp x) {
    return __imf_atanh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atanh (Tp x) {
        return __imf_atanhf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atanh (Tp x) {
        return __imf_atanh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atanh (Tp x) {
        return __imf_atanhf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atanh (Tp x) {
        return __imf_atanh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> atanh (Tp x) {
        return __imf_atanhf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> atanh (Tp x) {
        return __imf_atanh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// cbrt(x) function
/// calculate the cube root of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_cbrtf (float x);
double __imf_cbrt (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cbrt (Tp x) {
    return __imf_cbrtf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cbrt (Tp x) {
    return __imf_cbrt (x);
}

/// --------------------------------------------------------------------------
/// cdfnorm(x) function (normcdf(x) different name)
/// calculate the standard normal cumulative distribution function
/// --------------------------------------------------------------------------
extern "C" {
float __imf_cdfnormf (float x);
double __imf_cdfnorm (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cdfnorm (Tp x) {
    return __imf_cdfnormf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> normcdf (Tp x) {
    return __imf_cdfnormf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cdfnorm (Tp x) {
    return __imf_cdfnorm (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> normcdf (Tp x) {
    return __imf_cdfnorm (x);
}

/// --------------------------------------------------------------------------
/// cdfnorminv(x) function (normcdfinv(x) different name)
/// calculate the inverse of the standard normal cumulative distribution function
/// --------------------------------------------------------------------------
extern "C" {
float __imf_cdfnorminvf (float x);
double __imf_cdfnorminv (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cdfnorminv (Tp x) {
    return __imf_cdfnorminvf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> normcdfinv (Tp x) {
    return __imf_cdfnorminvf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cdfnorminv (Tp x) {
    return __imf_cdfnorminv (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> normcdfinv (Tp x) {
    return __imf_cdfnorminv (x);
}

/// --------------------------------------------------------------------------
/// cos(x) function
/// calculate the cosine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_cosf16 (_iml_half_internal x);
float __imf_cosf (float x);
float __imf_cosf_ha (float x);
float __imf_cosf_la (float x);
float __imf_cosf_ep (float x);
double __imf_cos (double x);
double __imf_cos_ha (double x);
double __imf_cos_la (double x);
double __imf_cos_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> cos (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_cosf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> cos(Tp x) {
    return sycl::half2{cos(x.s0()), cos(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cos (Tp x) {
    return __imf_cosf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cos (Tp x) {
    return __imf_cos (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cos (Tp x) {
        return __imf_cosf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cos (Tp x) {
        return __imf_cos_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cos (Tp x) {
        return __imf_cosf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cos (Tp x) {
        return __imf_cos_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cos (Tp x) {
        return __imf_cosf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cos (Tp x) {
        return __imf_cos_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// cosh(x) function
/// calculate the hyperbolic cosine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_coshf (float x);
float __imf_coshf_ha (float x);
float __imf_coshf_la (float x);
float __imf_coshf_ep (float x);
double __imf_cosh (double x);
double __imf_cosh_ha (double x);
double __imf_cosh_la (double x);
double __imf_cosh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cosh (Tp x) {
    return __imf_coshf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cosh (Tp x) {
    return __imf_cosh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cosh (Tp x) {
        return __imf_coshf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cosh (Tp x) {
        return __imf_cosh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cosh (Tp x) {
        return __imf_coshf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cosh (Tp x) {
        return __imf_cosh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cosh (Tp x) {
        return __imf_coshf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cosh (Tp x) {
        return __imf_cosh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// cospi(x) function
/// calculate the cosine of multiplied by pi input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_cospif (float x);
double __imf_cospi (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cospi (Tp x) {
    return __imf_cospif (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cospi (Tp x) {
    return __imf_cospi (x);
}

/// --------------------------------------------------------------------------
/// erf(x) function
/// calculate the error function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_erff (float x);
float __imf_erff_ha (float x);
float __imf_erff_la (float x);
float __imf_erff_ep (float x);
double __imf_erf (double x);
double __imf_erf_ha (double x);
double __imf_erf_la (double x);
double __imf_erf_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erf (Tp x) {
    return __imf_erff (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erf (Tp x) {
    return __imf_erf (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erf (Tp x) {
        return __imf_erff_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erf (Tp x) {
        return __imf_erf_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erf (Tp x) {
        return __imf_erff_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erf (Tp x) {
        return __imf_erf_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erf (Tp x) {
        return __imf_erff_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erf (Tp x) {
        return __imf_erf_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// erfc(x) function
/// calculate the complementary error function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_erfcf (float x);
float __imf_erfcf_ha (float x);
float __imf_erfcf_la (float x);
float __imf_erfcf_ep (float x);
double __imf_erfc (double x);
double __imf_erfc_ha (double x);
double __imf_erfc_la (double x);
double __imf_erfc_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfc (Tp x) {
    return __imf_erfcf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfc (Tp x) {
    return __imf_erfc (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfc (Tp x) {
        return __imf_erfcf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfc (Tp x) {
        return __imf_erfc_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfc (Tp x) {
        return __imf_erfcf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfc (Tp x) {
        return __imf_erfc_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfc (Tp x) {
        return __imf_erfcf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfc (Tp x) {
        return __imf_erfc_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// erfcinv(x) function
/// calculate the inverse complementary error function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_erfcinvf (float x);
double __imf_erfcinv (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfcinv (Tp x) {
    return __imf_erfcinvf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfcinv (Tp x) {
    return __imf_erfcinv (x);
}

/// --------------------------------------------------------------------------
/// erfcx(x) function
/// calculate the scaled complementary error function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_erfcxf (float x);
double __imf_erfcx (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfcx (Tp x) {
    return __imf_erfcxf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfcx (Tp x) {
    return __imf_erfcx (x);
}

/// --------------------------------------------------------------------------
/// erfinv(x) function
/// calculate the inverse error function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_erfinvf (float x);
double __imf_erfinv (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> erfinv (Tp x) {
    return __imf_erfinvf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> erfinv (Tp x) {
    return __imf_erfinv (x);
}

/// --------------------------------------------------------------------------
/// exp(x) function
/// calculate the base e exponential of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_expf16 (_iml_half_internal x);
_iml_half_internal __imf_expf16_ha (_iml_half_internal x);
_iml_half_internal __imf_expf16_la (_iml_half_internal x);
_iml_half_internal __imf_expf16_ep (_iml_half_internal x);
float __imf_expf (float x);
float __imf_expf_ha (float x);
float __imf_expf_la (float x);
float __imf_expf_ep (float x);
double __imf_exp (double x);
double __imf_exp_ha (double x);
double __imf_exp_la (double x);
double __imf_exp_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_expf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp(Tp x) {
    return sycl::half2{exp(x.s0()), exp(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp (Tp x) {
    return __imf_expf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp (Tp x) {
    return __imf_exp (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_expf16_ha (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp(Tp x) {
        return sycl::half2{exp(x.s0()), exp(x.s1())};
    }	
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp (Tp x) {
        return __imf_expf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp (Tp x) {
        return __imf_exp_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_expf16_la (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp(Tp x) {
        return sycl::half2{exp(x.s0()), exp(x.s1())};
    }	
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp (Tp x) {
        return __imf_expf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp (Tp x) {
        return __imf_exp_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_expf16_ep (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp(Tp x) {
        return sycl::half2{exp(x.s0()), exp(x.s1())};
    }	
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp (Tp x) {
        return __imf_expf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp (Tp x) {
        return __imf_exp_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// exp10(x) function
/// calculate the base 10 exponential of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_exp10f16 (_iml_half_internal x);
float __imf_exp10f (float x);
float __imf_exp10f_ha (float x);
float __imf_exp10f_la (float x);
float __imf_exp10f_ep (float x);
double __imf_exp10 (double x);
double __imf_exp10_ha (double x);
double __imf_exp10_la (double x);
double __imf_exp10_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp10 (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_exp10f16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp10(Tp x) {
    return sycl::half2{exp10(x.s0()), exp10(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp10 (Tp x) {
    return __imf_exp10f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp10 (Tp x) {
    return __imf_exp10 (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp10 (Tp x) {
        return __imf_exp10f_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp10 (Tp x) {
        return __imf_exp10_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp10 (Tp x) {
        return __imf_exp10f_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp10 (Tp x) {
        return __imf_exp10_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp10 (Tp x) {
        return __imf_exp10f_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp10 (Tp x) {
        return __imf_exp10_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// exp2(x) function
/// calculate the base 2 exponential of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_exp2f16 (_iml_half_internal x);
float __imf_exp2f (float x);
float __imf_exp2f_ha (float x);
float __imf_exp2f_la (float x);
float __imf_exp2f_ep (float x);
double __imf_exp2 (double x);
double __imf_exp2_ha (double x);
double __imf_exp2_la (double x);
double __imf_exp2_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> exp2 (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_exp2f16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> exp2(Tp x) {
    return sycl::half2{exp2(x.s0()), exp2(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp2 (Tp x) {
    return __imf_exp2f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp2 (Tp x) {
    return __imf_exp2 (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp2 (Tp x) {
        return __imf_exp2f_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp2 (Tp x) {
        return __imf_exp2_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp2 (Tp x) {
        return __imf_exp2f_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp2 (Tp x) {
        return __imf_exp2_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> exp2 (Tp x) {
        return __imf_exp2f_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> exp2 (Tp x) {
        return __imf_exp2_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// expm1(x) function
/// calculate the base exponential of the input argument, minus 1
/// --------------------------------------------------------------------------
extern "C" {
float __imf_expm1f (float x);
float __imf_expm1f_ha (float x);
float __imf_expm1f_la (float x);
float __imf_expm1f_ep (float x);
double __imf_expm1 (double x);
double __imf_expm1_ha (double x);
double __imf_expm1_la (double x);
double __imf_expm1_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> expm1 (Tp x) {
    return __imf_expm1f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> expm1 (Tp x) {
    return __imf_expm1 (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> expm1 (Tp x) {
        return __imf_expm1f_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> expm1 (Tp x) {
        return __imf_expm1_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> expm1 (Tp x) {
        return __imf_expm1f_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> expm1 (Tp x) {
        return __imf_expm1_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> expm1 (Tp x) {
        return __imf_expm1f_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> expm1 (Tp x) {
        return __imf_expm1_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// fdim(x, y) function
/// calculate the positive difference between x and y
/// --------------------------------------------------------------------------
extern "C" {
float __imf_fdimf (float x, float y);
double __imf_fdim (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> fdim (Tp x, Tp y) {
    return __imf_fdimf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> fdim (Tp x, Tp y) {
    return __imf_fdim (x, y);
}

/// --------------------------------------------------------------------------
/// fmod(x, y) function
/// calculate the floating-point remainder of x/y
/// --------------------------------------------------------------------------
extern "C" {
float __imf_fmodf (float x, float y);
double __imf_fmod (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> fmod (Tp x, Tp y) {
    return __imf_fmodf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> fmod (Tp x, Tp y) {
    return __imf_fmod (x, y);
}

/// --------------------------------------------------------------------------
/// frexp(x, r) function
/// extract mantissa and exponent of a floating-point value
/// --------------------------------------------------------------------------
extern "C" {
float __imf_frexpf (float x, int* r);
double __imf_frexp (double x, int* r);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> frexp (Tp x, int* r) {
    return __imf_frexpf (x, r);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> frexp (Tp x, int* r) {
    return __imf_frexp (x, r);
}

/// --------------------------------------------------------------------------
/// hypot(x, y) function
/// calculate the square root of the sum of squares of two arguments
/// --------------------------------------------------------------------------
extern "C" {
float __imf_hypotf (float x, float y);
float __imf_hypotf_ha (float x, float y);
float __imf_hypotf_la (float x, float y);
float __imf_hypotf_ep (float x, float y);
double __imf_hypot (double x, double y);
double __imf_hypot_ha (double x, double y);
double __imf_hypot_la (double x, double y);
double __imf_hypot_ep (double x, double y);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> hypot (Tp x, Tp y) {
    return __imf_hypotf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> hypot (Tp x, Tp y) {
    return __imf_hypot (x, y);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> hypot (Tp x, Tp y) {
        return __imf_hypotf_ha (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> hypot (Tp x, Tp y) {
        return __imf_hypot_ha (x, y);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> hypot (Tp x, Tp y) {
        return __imf_hypotf_la (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> hypot (Tp x, Tp y) {
        return __imf_hypot_la (x, y);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> hypot (Tp x, Tp y) {
        return __imf_hypotf_ep (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> hypot (Tp x, Tp y) {
        return __imf_hypot_ep (x, y);
    }
}

/// --------------------------------------------------------------------------
/// cyl_bessel_i0(x) function
/// calculate the value of the regular modified cylindrical Bessel function of order 0 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_i0f (float x);
double __imf_i0 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cyl_bessel_i0 (Tp x) {
    return __imf_i0f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cyl_bessel_i0 (Tp x) {
    return __imf_i0 (x);
}

/// --------------------------------------------------------------------------
/// cyl_bessel_i1(x) function
/// calculate the value of the regular modified cylindrical Bessel function of order 1 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_i1f (float x);
double __imf_i1 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> cyl_bessel_i1 (Tp x) {
    return __imf_i1f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> cyl_bessel_i1 (Tp x) {
    return __imf_i1 (x);
}

/// --------------------------------------------------------------------------
/// ilogb(x) function
/// compute the unbiased integer exponent of the argument
/// --------------------------------------------------------------------------
extern "C" {
int __imf_ilogbf (float x);
int __imf_ilogb (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, int> ilogb (Tp x) {
    return __imf_ilogbf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, int> ilogb (Tp x) {
    return __imf_ilogb (x);
}

/// --------------------------------------------------------------------------
/// isfinite(x) function
/// determine whether argument is finite
/// --------------------------------------------------------------------------
extern "C" {
int __imf_isfinitef (float x);
int __imf_isfinite (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, int> isfinite (Tp x) {
    return __imf_isfinitef (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, int> isfinite (Tp x) {
    return __imf_isfinite (x);
}

/// --------------------------------------------------------------------------
/// isinf(x) function
/// determine whether argument is infinite
/// --------------------------------------------------------------------------
extern "C" {
int __imf_isinff (float x);
int __imf_isinf (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, int> isinf (Tp x) {
    return __imf_isinff (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, int> isinf (Tp x) {
    return __imf_isinf (x);
}

/// --------------------------------------------------------------------------
/// isnan(x) function
/// determine whether argument is a nan
/// --------------------------------------------------------------------------
extern "C" {
int __imf_isnanf (float x);
int __imf_isnan (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, int> isnan (Tp x) {
    return __imf_isnanf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, int> isnan (Tp x) {
    return __imf_isnan (x);
}

/// --------------------------------------------------------------------------
/// j0(x) function
/// calculate the value of the Bessel function of the first kind of order 0 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_j0f (float x);
double __imf_j0 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> j0 (Tp x) {
    return __imf_j0f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> j0 (Tp x) {
    return __imf_j0 (x);
}

/// --------------------------------------------------------------------------
/// j1(x) function
/// calculate the value of the Bessel function of the first kind of order 1 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_j1f (float x);
double __imf_j1 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> j1 (Tp x) {
    return __imf_j1f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> j1 (Tp x) {
    return __imf_j1 (x);
}

/// --------------------------------------------------------------------------
/// jn(n, x) function
/// calculate the value of the Bessel function of the first kind of order n for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_jnf (int n, float x);
double __imf_jn (int n, double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> jn (int n, Tp x) {
    return __imf_jnf (n, x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> jn (int n, Tp x) {
    return __imf_jn (n, x);
}

/// --------------------------------------------------------------------------
/// ldexp(x, y) function
/// calculate the value of x*2^y
/// --------------------------------------------------------------------------
extern "C" {
float __imf_ldexpf (float x, int y);
double __imf_ldexp (double x, int y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> ldexp (Tp x, int y) {
    return __imf_ldexpf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> ldexp (Tp x, int y) {
    return __imf_ldexp (x, y);
}

/// --------------------------------------------------------------------------
/// lgamma(x) function
/// calculate the natural logarithm of the absolute value of the gamma function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_lgammaf (float x);
double __imf_lgamma (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> lgamma (Tp x) {
    return __imf_lgammaf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> lgamma (Tp x) {
    return __imf_lgamma (x);
}

/// --------------------------------------------------------------------------
/// llrint(x) function
/// round input to nearest integer value
/// --------------------------------------------------------------------------
extern "C" {
long long int __imf_llrintf (float x);
long long int __imf_llrint (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, long long int> llrint (Tp x) {
    return __imf_llrintf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, long long int> llrint (Tp x) {
    return __imf_llrint (x);
}

/// --------------------------------------------------------------------------
/// llround(x) function
/// round to nearest integer value
/// --------------------------------------------------------------------------
extern "C" {
long long int __imf_llroundf (float x);
long long int __imf_llround (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, long long int> llround (Tp x) {
    return __imf_llroundf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, long long int> llround (Tp x) {
    return __imf_llround (x);
}

/// --------------------------------------------------------------------------
/// log(x) function
/// calculate the natural logarithm of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_logf16 (_iml_half_internal x);
_iml_half_internal __imf_logf16_ha (_iml_half_internal x);
_iml_half_internal __imf_logf16_la (_iml_half_internal x);
_iml_half_internal __imf_logf16_ep (_iml_half_internal x);
float __imf_logf (float x);
float __imf_logf_ha (float x);
float __imf_logf_la (float x);
float __imf_logf_ep (float x);
double __imf_log (double x);
double __imf_log_ha (double x);
double __imf_log_la (double x);
double __imf_log_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_logf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log (Tp x) {
    return __imf_logf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log (Tp x) {
    return __imf_log (x);
}
template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log(Tp x) {
    return sycl::half2{log(x.s0()), log(x.s1())};
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_logf16_ha (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log(Tp x) {
        return sycl::half2{log(x.s0()), log(x.s1())};
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log (Tp x) {
        return __imf_logf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log (Tp x) {
        return __imf_log_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_logf16_la (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log(Tp x) {
        return sycl::half2{log(x.s0()), log(x.s1())};
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log (Tp x) {
        return __imf_logf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log (Tp x) {
        return __imf_log_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log (Tp x) {
        _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
        return sycl::bit_cast<sycl::half>(__imf_logf16_ep (xi));
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log(Tp x) {
        return sycl::half2{log(x.s0()), log(x.s1())};
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log (Tp x) {
        return __imf_logf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log (Tp x) {
        return __imf_log_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// log10(x) function
/// calculate the base 10 logarithm of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_log10f16 (_iml_half_internal x);
float __imf_log10f (float x);
float __imf_log10f_ha (float x);
float __imf_log10f_la (float x);
float __imf_log10f_ep (float x);
double __imf_log10 (double x);
double __imf_log10_ha (double x);
double __imf_log10_la (double x);
double __imf_log10_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log10 (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_log10f16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log10(Tp x) {
    return sycl::half2{log10(x.s0()), log10(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log10 (Tp x) {
    return __imf_log10f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log10 (Tp x) {
    return __imf_log10 (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log10 (Tp x) {
        return __imf_log10f_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log10 (Tp x) {
        return __imf_log10_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log10 (Tp x) {
        return __imf_log10f_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log10 (Tp x) {
        return __imf_log10_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log10 (Tp x) {
        return __imf_log10f_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log10 (Tp x) {
        return __imf_log10_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// log1p(x) function
/// calculate the value of log(1+x)
/// --------------------------------------------------------------------------
extern "C" {
float __imf_log1pf (float x);
float __imf_log1pf_ha (float x);
float __imf_log1pf_la (float x);
float __imf_log1pf_ep (float x);
double __imf_log1p (double x);
double __imf_log1p_ha (double x);
double __imf_log1p_la (double x);
double __imf_log1p_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log1p (Tp x) {
    return __imf_log1pf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log1p (Tp x) {
    return __imf_log1p (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log1p (Tp x) {
        return __imf_log1pf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log1p (Tp x) {
        return __imf_log1p_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log1p (Tp x) {
        return __imf_log1pf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log1p (Tp x) {
        return __imf_log1p_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log1p (Tp x) {
        return __imf_log1pf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log1p (Tp x) {
        return __imf_log1p_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// log2(x) function
/// calculate the base 2 logarithm of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_log2f16 (_iml_half_internal x);
float __imf_log2f (float x);
float __imf_log2f_ha (float x);
float __imf_log2f_la (float x);
float __imf_log2f_ep (float x);
double __imf_log2 (double x);
double __imf_log2_ha (double x);
double __imf_log2_la (double x);
double __imf_log2_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> log2 (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_log2f16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> log2(Tp x) {
    return sycl::half2{log2(x.s0()), log2(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log2 (Tp x) {
    return __imf_log2f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log2 (Tp x) {
    return __imf_log2 (x);
}

// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log2 (Tp x) {
        return __imf_log2f_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log2 (Tp x) {
        return __imf_log2_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log2 (Tp x) {
        return __imf_log2f_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log2 (Tp x) {
        return __imf_log2_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> log2 (Tp x) {
        return __imf_log2f_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> log2 (Tp x) {
        return __imf_log2_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// logb(x) function
/// calculate the floating-point representation of the exponent of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_logbf (float x);
double __imf_logb (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> logb (Tp x) {
    return __imf_logbf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> logb (Tp x) {
    return __imf_logb (x);
}

/// --------------------------------------------------------------------------
/// lrint(x) function
/// round input to nearest integer value
/// --------------------------------------------------------------------------
extern "C" {
long int __imf_lrintf (float x);
long int __imf_lrint (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, long int> lrint (Tp x) {
    return __imf_lrintf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, long int> lrint (Tp x) {
    return __imf_lrint (x);
}

/// --------------------------------------------------------------------------
/// lround(x) function
/// round to nearest integer value
/// --------------------------------------------------------------------------
extern "C" {
long int __imf_lroundf (float x);
long int __imf_lround (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, long int> lround (Tp x) {
    return __imf_lroundf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, long int> lround (Tp x) {
    return __imf_lround (x);
}

/// --------------------------------------------------------------------------
/// modf(x, r) function
/// break down the input argument into fractional and integral parts
/// --------------------------------------------------------------------------
extern "C" {
float __imf_modff (float x, float* r);
double __imf_modf (double x, double* r);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> modf (Tp x, Tp* r) {
    return __imf_modff (x, r);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> modf (Tp x, Tp* r) {
    return __imf_modf (x, r);
}

/// --------------------------------------------------------------------------
/// nan(x) function
/// return "not a number" value
/// --------------------------------------------------------------------------
extern "C" {
float __imf_nanf (const char* x);
double __imf_nan (const char* x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, const char*>, float> nanf (Tp x) {
    return __imf_nanf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, const char*>, double> nan (Tp x) {
    return __imf_nan (x);
}

/// --------------------------------------------------------------------------
/// nextafter(x, y) function
/// return next representable floating-point value after argument x in the direction of y
/// --------------------------------------------------------------------------
extern "C" {
float __imf_nextafterf (float x, float y);
double __imf_nextafter (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> nextafter (Tp x, Tp y) {
    return __imf_nextafterf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> nextafter (Tp x, Tp y) {
    return __imf_nextafter (x, y);
}

/// --------------------------------------------------------------------------
/// norm(n, x) function
/// calculate the square root of the sum of squares of any number of coordinates
/// --------------------------------------------------------------------------
extern "C" {
float __imf_normf (int n, const float* x);
double __imf_norm (int n, const double* x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> norm (int n, const Tp* x) {
    return __imf_normf (n, x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> norm (int n, const Tp* x) {
    return __imf_norm (n, x);
}

/// --------------------------------------------------------------------------
/// norm3d(x, y, z) function
/// calculate the square root of the sum of squares of three coordinates of the argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_norm3df (float x, float y, float z);
double __imf_norm3d (double x, double y, double z);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> norm3d (Tp x, Tp y, Tp z) {
    return __imf_norm3df (x, y, z);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> norm3d (Tp x, Tp y, Tp z) {
    return __imf_norm3d (x, y, z);
}

/// --------------------------------------------------------------------------
/// norm4d(x, y, z, w) function
/// calculate the square root of the sum of squares of four coordinates of the argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_norm4df (float x, float y, float z, float w);
double __imf_norm4d (double x, double y, double z, double w);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> norm4d (Tp x, Tp y, Tp z, Tp w) {
    return __imf_norm4df (x, y, z, w);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> norm4d (Tp x, Tp y, Tp z, Tp w) {
    return __imf_norm4d (x, y, z, w);
}

/// --------------------------------------------------------------------------
/// pow(x, y) function
/// calculate the value of first argument to the power of second argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_powf (float x, float y);
float __imf_powf_ha (float x, float y);
float __imf_powf_la (float x, float y);
float __imf_powf_ep (float x, float y);
double __imf_pow (double x, double y);
double __imf_pow_ha (double x, double y);
double __imf_pow_la (double x, double y);
double __imf_pow_ep (double x, double y);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> pow (Tp x, Tp y) {
    return __imf_powf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> pow (Tp x, Tp y) {
    return __imf_pow (x, y);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> pow (Tp x, Tp y) {
        return __imf_powf_ha (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> pow (Tp x, Tp y) {
        return __imf_pow_ha (x, y);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> pow (Tp x, Tp y) {
        return __imf_powf_la (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> pow (Tp x, Tp y) {
        return __imf_pow_la (x, y);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> pow (Tp x, Tp y) {
        return __imf_powf_ep (x, y);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> pow (Tp x, Tp y) {
        return __imf_pow_ep (x, y);
    }
}

/// --------------------------------------------------------------------------
/// powi(x, y) function
/// calculate the value of first argument to the integer power of second argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_pownf (float x, int y);
double __imf_pown (double x, int y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> powi (Tp x, int y) {
    return __imf_pownf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> powi (Tp x, int y) {
    return __imf_pown (x, y);
}

/// --------------------------------------------------------------------------
/// rcbrt(x) function
/// calculate reciprocal cube root function
/// --------------------------------------------------------------------------
extern "C" {
float __imf_rcbrtf (float x);
double __imf_rcbrt (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rcbrt (Tp x) {
    return __imf_rcbrtf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rcbrt (Tp x) {
    return __imf_rcbrt (x);
}

/// --------------------------------------------------------------------------
/// remainder(x, y) function
/// calculate floating-point remainder
/// --------------------------------------------------------------------------
extern "C" {
float __imf_remainderf (float x, float y);
double __imf_remainder (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> remainder (Tp x, Tp y) {
    return __imf_remainderf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> remainder (Tp x, Tp y) {
    return __imf_remainder (x, y);
}

/// --------------------------------------------------------------------------
/// remquo(x, y, q) function
/// compute floating-point remainder and part of quotient
/// --------------------------------------------------------------------------
extern "C" {
float __imf_remquof (float x, float y, int* q);
double __imf_remquo (double x, double y, int* q);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> remquo (Tp x, Tp y, int* q) {
    return __imf_remquof (x, y, q);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> remquo (Tp x, Tp y, int* q) {
    return __imf_remquo (x, y, q);
}

/// --------------------------------------------------------------------------
/// rhypot(x, y) function
/// calculate one over the square root of the sum of squares of two arguments
/// --------------------------------------------------------------------------
extern "C" {
float __imf_rhypotf (float x, float y);
double __imf_rhypot (double x, double y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rhypot (Tp x, Tp y) {
    return __imf_rhypotf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rhypot (Tp x, Tp y) {
    return __imf_rhypot (x, y);
}

/// --------------------------------------------------------------------------
/// rnorm(n, x) function
/// calculate the reciprocal of square root of the sum of squares of any number of coordinates
/// --------------------------------------------------------------------------
extern "C" {
float __imf_rnormf (int n, const float* x);
double __imf_rnorm (int n, const double* x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rnorm (int n, const Tp* x) {
    return __imf_rnormf (n, x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rnorm (int n, const Tp* x) {
    return __imf_rnorm (n, x);
}

/// --------------------------------------------------------------------------
/// rnorm3d(x, y, z) function
/// calculate one over the square root of the sum of squares of three coordinates
/// --------------------------------------------------------------------------
extern "C" {
float __imf_rnorm3df (float x, float y, float z);
double __imf_rnorm3d (double x, double y, double z);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rnorm3d (Tp x, Tp y, Tp z) {
    return __imf_rnorm3df (x, y, z);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rnorm3d (Tp x, Tp y, Tp z) {
    return __imf_rnorm3d (x, y, z);
}

/// --------------------------------------------------------------------------
/// rnorm4d(x, y, z, w) function
/// calculate one over the square root of the sum of squares of four coordinates
/// --------------------------------------------------------------------------
extern "C" {
float __imf_rnorm4df (float x, float y, float z, float w);
double __imf_rnorm4d (double x, double y, double z, double w);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> rnorm4d (Tp x, Tp y, Tp z, Tp w) {
    return __imf_rnorm4df (x, y, z, w);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> rnorm4d (Tp x, Tp y, Tp z, Tp w) {
    return __imf_rnorm4d (x, y, z, w);
}

/// --------------------------------------------------------------------------
/// round(x) function
/// round to nearest integer value in floating-point
/// --------------------------------------------------------------------------
extern "C" {
float __imf_roundf (float x);
double __imf_round (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> round (Tp x) {
    return __imf_roundf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> round (Tp x) {
    return __imf_round (x);
}

/// --------------------------------------------------------------------------
/// scalbn(x, y) function
/// scale floating-point input by integer power of two
/// --------------------------------------------------------------------------
extern "C" {
float __imf_scalbnf (float x, int y);
double __imf_scalbn (double x, int y);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> scalbn (Tp x, int y) {
    return __imf_scalbnf (x, y);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> scalbn (Tp x, int y) {
    return __imf_scalbn (x, y);
}

/// --------------------------------------------------------------------------
/// signbit(x) function
/// return the sign bit of the input
/// --------------------------------------------------------------------------
extern "C" {
int __imf_signbitf (float x);
int __imf_signbit (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, int> signbit (Tp x) {
    return __imf_signbitf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, int> signbit (Tp x) {
    return __imf_signbit (x);
}

/// --------------------------------------------------------------------------
/// sin(x) function
/// calculate the sine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
_iml_half_internal __imf_sinf16 (_iml_half_internal x);
float __imf_sinf (float x);
float __imf_sinf_ha (float x);
float __imf_sinf_la (float x);
float __imf_sinf_ep (float x);
double __imf_sin (double x);
double __imf_sin_ha (double x);
double __imf_sin_la (double x);
double __imf_sin_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half>, sycl::half> sin (Tp x) {
    _iml_half_internal xi = sycl::bit_cast<_iml_half_internal>(x);
    return sycl::bit_cast<sycl::half>(__imf_sinf16 (xi));
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, sycl::half2>, sycl::half2> sin(Tp x) {
    return sycl::half2{sin(x.s0()), sin(x.s1())};
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sin (Tp x) {
    return __imf_sinf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sin (Tp x) {
    return __imf_sin (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sin (Tp x) {
        return __imf_sinf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sin (Tp x) {
        return __imf_sin_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sin (Tp x) {
        return __imf_sinf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sin (Tp x) {
        return __imf_sin_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sin (Tp x) {
        return __imf_sinf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sin (Tp x) {
        return __imf_sin_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// sincos(x, s, c) function
/// calculate the sine and cosine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
void __imf_sincosf (float x, float* s, float* c);
void __imf_sincosf_ha (float x, float* s, float* c);
void __imf_sincosf_la (float x, float* s, float* c);
void __imf_sincosf_ep (float x, float* s, float* c);
void __imf_sincos (double x, double* s, double* c);
void __imf_sincos_ha (double x, double* s, double* c);
void __imf_sincos_la (double x, double* s, double* c);
void __imf_sincos_ep (double x, double* s, double* c);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, void> sincos (Tp x, Tp* s, Tp* c) {
    return __imf_sincosf (x, s, c);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, void> sincos (Tp x, Tp* s, Tp* c) {
    return __imf_sincos (x, s, c);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincosf_ha (x, s, c);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincos_ha (x, s, c);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincosf_la (x, s, c);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincos_la (x, s, c);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincosf_ep (x, s, c);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, void> sincos (Tp x, Tp* s, Tp* c) {
        return __imf_sincos_ep (x, s, c);
    }
}

/// --------------------------------------------------------------------------
/// sincospi(x, s, c) function
/// calculate the sine and cosine of multiplied by pi input argument
/// --------------------------------------------------------------------------
extern "C" {
void __imf_sincospif (float x, float* s, float* c);
void __imf_sincospi (double x, double* s, double* c);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, void> sincospi (Tp x, Tp* s, Tp* c) {
    return __imf_sincospif (x, s, c);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, void> sincospi (Tp x, Tp* s, Tp* c) {
    return __imf_sincospi (x, s, c);
}

/// --------------------------------------------------------------------------
/// sinh(x) function
/// calculate the hyperbolic sine of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_sinhf (float x);
float __imf_sinhf_ha (float x);
float __imf_sinhf_la (float x);
float __imf_sinhf_ep (float x);
double __imf_sinh (double x);
double __imf_sinh_ha (double x);
double __imf_sinh_la (double x);
double __imf_sinh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sinh (Tp x) {
    return __imf_sinhf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sinh (Tp x) {
    return __imf_sinh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sinh (Tp x) {
        return __imf_sinhf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sinh (Tp x) {
        return __imf_sinh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sinh (Tp x) {
        return __imf_sinhf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sinh (Tp x) {
        return __imf_sinh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sinh (Tp x) {
        return __imf_sinhf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sinh (Tp x) {
        return __imf_sinh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// sinpi(x) function
/// calculate the sine of multiplied by pi input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_sinpif (float x);
double __imf_sinpi (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> sinpi (Tp x) {
    return __imf_sinpif (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> sinpi (Tp x) {
    return __imf_sinpi (x);
}

/// --------------------------------------------------------------------------
/// tan(x) function
/// calculate the tangent of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_tanf (float x);
float __imf_tanf_ha (float x);
float __imf_tanf_la (float x);
float __imf_tanf_ep (float x);
double __imf_tan (double x);
double __imf_tan_ha (double x);
double __imf_tan_la (double x);
double __imf_tan_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tan (Tp x) {
    return __imf_tanf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tan (Tp x) {
    return __imf_tan (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tan (Tp x) {
        return __imf_tanf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tan (Tp x) {
        return __imf_tan_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tan (Tp x) {
        return __imf_tanf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tan (Tp x) {
        return __imf_tan_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tan (Tp x) {
        return __imf_tanf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tan (Tp x) {
        return __imf_tan_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// tanh(x) function
/// calculate the hyperbolic tangent of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_tanhf (float x);
float __imf_tanhf_ha (float x);
float __imf_tanhf_la (float x);
float __imf_tanhf_ep (float x);
double __imf_tanh (double x);
double __imf_tanh_ha (double x);
double __imf_tanh_la (double x);
double __imf_tanh_ep (double x);
};

//   default accuracy
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tanh (Tp x) {
    return __imf_tanhf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tanh (Tp x) {
    return __imf_tanh (x);
}
// high accuracy
namespace ha {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tanh (Tp x) {
        return __imf_tanhf_ha (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tanh (Tp x) {
        return __imf_tanh_ha (x);
    }
}
// low accuracy
namespace la {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tanh (Tp x) {
        return __imf_tanhf_la (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tanh (Tp x) {
        return __imf_tanh_la (x);
    }
}
// enhanced performance
namespace ep {
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tanh (Tp x) {
        return __imf_tanhf_ep (x);
    }
    template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tanh (Tp x) {
        return __imf_tanh_ep (x);
    }
}

/// --------------------------------------------------------------------------
/// tgamma(x) function
/// calculate the gamma function of the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_tgammaf (float x);
double __imf_tgamma (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> tgamma (Tp x) {
    return __imf_tgammaf (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> tgamma (Tp x) {
    return __imf_tgamma (x);
}

/// --------------------------------------------------------------------------
/// y0(x) function
/// calculate the value of the Bessel function of the second kind of order 0 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_y0f (float x);
double __imf_y0 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> y0 (Tp x) {
    return __imf_y0f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> y0 (Tp x) {
    return __imf_y0 (x);
}

/// --------------------------------------------------------------------------
/// y1(x) function
/// calculate the value of the Bessel function of the second kind of order 1 for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_y1f (float x);
double __imf_y1 (double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> y1 (Tp x) {
    return __imf_y1f (x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> y1 (Tp x) {
    return __imf_y1 (x);
}

/// --------------------------------------------------------------------------
/// yn(n, x) function
/// calculate the value of the Bessel function of the second kind of order n for the input argument
/// --------------------------------------------------------------------------
extern "C" {
float __imf_ynf (int n, float x);
double __imf_yn (int n, double x);
};

template <typename Tp> std::enable_if_t<std::is_same_v<Tp, float>, float> yn (int n, Tp x) {
    return __imf_ynf (n, x);
}
template <typename Tp> std::enable_if_t<std::is_same_v<Tp, double>, double> yn (int n, Tp x) {
    return __imf_yn (n, x);
}

/// --------------------------------------------------------------------------
/// rcp64h(x) function
/// calculate upper 32 bits of double reciprocal
/// --------------------------------------------------------------------------
extern "C" {
double __imf_rcp64h (double x);
};

template <typename Tp>
std::enable_if_t<std::is_same_v<Tp, double>, double> rcp64h(Tp x) {
  return __imf_rcp64h(x);
}


} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
