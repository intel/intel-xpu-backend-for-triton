// -*- C++ -*-
//
// Modifications, Copyright (C) 2022 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may not
// use, modify, copy, publish, distribute, disclose or transmit this software or
// the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//
//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <sycl/detail/builtins/builtins.hpp>

#ifdef __SYCL_DEVICE_ONLY__
extern "C" {

extern __DPCPP_SYCL_EXTERNAL void *memcpy(void *dest, const void *src,
                                          size_t n);
extern __DPCPP_SYCL_EXTERNAL void *memset(void *dest, int c, size_t n);
extern __DPCPP_SYCL_EXTERNAL int memcmp(const void *s1, const void *s2,
                                        size_t n);
extern __DPCPP_SYCL_EXTERNAL int rand();
extern __DPCPP_SYCL_EXTERNAL void srand(unsigned int seed);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmax(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llmin(long long int x,
                                                       long long int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_max(int x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_min(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmax(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_ullmin(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umax(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umin(unsigned int x,
                                                     unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_brev(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_brevll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_byte_perm(unsigned int x, unsigned int y, unsigned int s);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffs(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_ffsll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clz(int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_clzll(long long int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popc(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL int __imf_popcll(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_sad(int x, int y,
                                                    unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_usad(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_rhadd(int x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_hadd(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_urhadd(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_uhadd(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mul24(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umul24(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_mulhi(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_umulhi(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_mul64hi(long long int x,
                                                         long long int y);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_umul64hi(unsigned long long int x, unsigned long long int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_abs(int x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llabs(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_saturatef(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fabsf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_floorf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ceilf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_truncf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_nearbyintf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rsqrtf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_invf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaxf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fminf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_copysignf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_exp10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_expf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_logf(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log2f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_log10f(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_powf(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fast_fdividef(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fadd_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fsub_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmul_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rd(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rn(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_ru(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdiv_rz(float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rd(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rn(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_ru(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_frcp_rz(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rd(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rn(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_ru(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmaf_rz(float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rd(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rn(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_ru(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sqrtf_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rd(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rn(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_ru(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float2int_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rz(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rd(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rn(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_ru(float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rz(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rd(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rn(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_ru(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_float_as_int(float x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_float_as_uint(float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rd(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rn(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_ru(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int2float_rz(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_int_as_float(int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ll2float_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint2float_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_uint_as_float(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ull2float_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rd(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rn(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_ru(float x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_float2half_rz(float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL int __imf_half2int_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half2short_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rd(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rn(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_ru(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rz(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL short __imf_half_as_short(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL unsigned short __imf_half_as_ushort(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rd(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rn(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_ru(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_int2half_rz(int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ll2half_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rd(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rn(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_ru(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short2half_rz(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_short_as_half(short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_uint2half_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ull2half_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort2half_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ushort_as_half(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_double2half(double x);

extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaf16(_Float16 x, _Float16 y,
                                                   _Float16 z);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fabsf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_floorf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_ceilf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_truncf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_nearbyintf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_sqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_rsqrtf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_invf16(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fmaxf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_fminf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_copysignf16(_Float16 x, _Float16 y);
extern __DPCPP_SYCL_EXTERNAL float __imf_half2float(_Float16 x);
extern __DPCPP_SYCL_EXTERNAL float __imf_bfloat162float(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat162ushort_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long
__imf_bfloat162ull_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rd(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rn(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_ru(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rz(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rd(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rn(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_ru(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rz(float x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rd(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rn(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_ru(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort2bfloat16_rz(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rd(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rn(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_ru(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ull2bfloat16_rz(unsigned long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rd(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rn(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_ru(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rz(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rd(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rn(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_ru(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rz(int x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rd(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rn(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_ru(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rz(long long x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_double2bfloat16(double x);
extern __DPCPP_SYCL_EXTERNAL short __imf_bfloat16_as_short(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL unsigned short
__imf_bfloat16_as_ushort(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_short_as_bfloat16(short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t
__imf_ushort_as_bfloat16(unsigned short x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmabf16(uint16_t x, uint16_t y,
                                                    uint16_t z);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fmaxbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fminbf16(uint16_t x, uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_fabsbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rintbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_floorbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_ceilbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_truncbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_copysignbf16(uint16_t x,
                                                         uint16_t y);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_sqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL uint16_t __imf_rsqrtbf16(uint16_t x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rd(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rn(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_ru(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fma_rz(double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_fabs(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_floor(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ceil(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_trunc(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rcp64h(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_nearbyint(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rsqrt(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_inv(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmax(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmin(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_copysign(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dadd_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dsub_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_dmul_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rd(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rn(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_ru(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_ddiv_rz(double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rd(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rn(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_ru(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_drcp_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rd(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rn(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_ru(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sqrt_rz(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rd(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rn(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_ru(double x);
extern __DPCPP_SYCL_EXTERNAL float __imf_double2float_rz(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2hiint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2loint(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rd(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rn(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_ru(double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_double2int_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_int2double_rn(int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rd(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rn(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_ru(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double2ll_rz(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rd(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rn(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_ru(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ll2double_rz(long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rd(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rn(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_ru(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL double
__imf_ull2double_rz(unsigned long long int x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rd(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rn(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_ru(double x);
extern __DPCPP_SYCL_EXTERNAL unsigned long long int
__imf_double2ull_rz(double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_double_as_longlong(double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_longlong_as_double(long long int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rd(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rn(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_ru(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_uint2double_rz(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL double __imf_hiloint2double(int hi, int lo);

extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabs4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vneg4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss2(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vnegss4(unsigned int x);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffs4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu2(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vabsdiffu4(unsigned int x,
                                                           unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vadd4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vaddus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub2(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsub4(unsigned int x,
                                                      unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubss4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsubus4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vavgu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vhaddu4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpeq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmples4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmplts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vcmpltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxs4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmaxu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vmins4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vminu4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vseteq4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne2(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetne4(unsigned int x,
                                                        unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetges4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgeu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetgtu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetles4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetleu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetlts4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu2(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsetltu4(unsigned int x,
                                                         unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsads4(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu2(unsigned int x,
                                                       unsigned int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vsadu4(unsigned int x,
                                                       unsigned int y);
/*
 * IMF math builtins
 */
// float16 imf builtins
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_cosf16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_exp10f16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_exp2f16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_expf16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_expf16_ha (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_expf16_la (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_expf16_ep (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_logf16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_logf16_ha (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_logf16_la (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_logf16_ep (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_log10f16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_log2f16 (_Float16 x);
extern __DPCPP_SYCL_EXTERNAL _Float16 __imf_sinf16 (_Float16 x);
// float32 imf builtins
extern __DPCPP_SYCL_EXTERNAL float __imf_acosf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acosf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acosf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acosf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acoshf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acoshf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acoshf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_acoshf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinhf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinhf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinhf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_asinhf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atan2f (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_atan2f_ep (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_atan2f_ha (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_atan2f_la (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanhf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanhf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanhf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_atanhf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cbrtf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cdfnormf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cdfnorminvf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cosf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cosf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cosf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cosf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_coshf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_coshf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_coshf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_coshf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_cospif (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erff (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erff_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erff_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erff_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcinvf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfcxf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_erfinvf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp10f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp10f_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp10f_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp10f_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp2f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp2f_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp2f_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_exp2f_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expm1f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expm1f_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expm1f_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_expm1f_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_fdimf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_fmodf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_frexpf (float x, int* r);
extern __DPCPP_SYCL_EXTERNAL float __imf_hypotf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_hypotf_ep (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_hypotf_ha (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_hypotf_la (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_i0f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_i1f (float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_ilogbf (float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isfinitef (float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isinff (float x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isnanf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_j0f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_j1f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_jnf (int n, float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ldexpf (float x, int y);
extern __DPCPP_SYCL_EXTERNAL float __imf_lgammaf (float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llrintf (float x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llroundf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_logf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_logf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_logf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_logf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log10f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log10f_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log10f_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log10f_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log1pf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log1pf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log1pf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log1pf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log2f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log2f_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log2f_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_log2f_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_logbf (float x);
extern __DPCPP_SYCL_EXTERNAL long int __imf_lrintf (float x);
extern __DPCPP_SYCL_EXTERNAL long int __imf_lroundf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_modff (float x, float* r);
extern __DPCPP_SYCL_EXTERNAL float __imf_nanf (const char* x);
extern __DPCPP_SYCL_EXTERNAL float __imf_nextafterf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_normf (int n, const float* x);
extern __DPCPP_SYCL_EXTERNAL float __imf_norm3df (float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_norm4df (float x, float y, float z, float w);
extern __DPCPP_SYCL_EXTERNAL float __imf_powf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_powf_ep (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_powf_ha (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_powf_la (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_pownf (float x, int y);
extern __DPCPP_SYCL_EXTERNAL float __imf_rcbrtf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_remainderf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_remquof (float x, float y, int* q);
extern __DPCPP_SYCL_EXTERNAL float __imf_rhypotf (float x, float y);
extern __DPCPP_SYCL_EXTERNAL float __imf_rnormf (int n, const float* x);
extern __DPCPP_SYCL_EXTERNAL float __imf_rnorm3df (float x, float y, float z);
extern __DPCPP_SYCL_EXTERNAL float __imf_rnorm4df (float x, float y, float z, float w);
extern __DPCPP_SYCL_EXTERNAL float __imf_roundf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_scalbnf (float x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_signbitf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinf_la (float x);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincosf (float x, float* s, float* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincosf_ep (float x, float* s, float* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincosf_ha (float x, float* s, float* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincosf_la (float x, float* s, float* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincospif (float x, float* s, float* c);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinhf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinhf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinhf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinhf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_sinpif (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanhf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanhf_ep (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanhf_ha (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tanhf_la (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_tgammaf (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_y0f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_y1f (float x);
extern __DPCPP_SYCL_EXTERNAL float __imf_ynf (int n, float x);
// float64 imf builtins
extern __DPCPP_SYCL_EXTERNAL double __imf_acos (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acos_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acos_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acos_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acosh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acosh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acosh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_acosh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asin (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asin_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asin_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asin_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asinh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asinh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asinh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_asinh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan2 (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan2_ep (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan2_ha (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_atan2_la (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_atanh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atanh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atanh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_atanh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cbrt (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cdfnorm (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cdfnorminv (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cos (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cos_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cos_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cos_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cosh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cosh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cosh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cosh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_cospi (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erf (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erf_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erf_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erf_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfc (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfc_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfc_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfc_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfcinv (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfcx (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_erfinv (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp10 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp10_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp10_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp10_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp2 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp2_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp2_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_exp2_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_expm1 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_expm1_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_expm1_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_expm1_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_fdim (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_fmod (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_frexp (double x, int* r);
extern __DPCPP_SYCL_EXTERNAL double __imf_hypot (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_hypot_ep (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_hypot_ha (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_hypot_la (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_i0 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_i1 (double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_ilogb (double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isfinite (double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isinf (double x);
extern __DPCPP_SYCL_EXTERNAL int __imf_isnan (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_j0 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_j1 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_jn (int n, double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_ldexp (double x, int y);
extern __DPCPP_SYCL_EXTERNAL double __imf_lgamma (double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llrint (double x);
extern __DPCPP_SYCL_EXTERNAL long long int __imf_llround (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log10 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log10_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log10_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log10_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log1p (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log1p_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log1p_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log1p_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log2 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log2_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log2_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_log2_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_logb (double x);
extern __DPCPP_SYCL_EXTERNAL long int __imf_lrint (double x);
extern __DPCPP_SYCL_EXTERNAL long int __imf_lround (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_modf (double x, double* r);
extern __DPCPP_SYCL_EXTERNAL double __imf_nan (const char* x);
extern __DPCPP_SYCL_EXTERNAL double __imf_nextafter (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_norm (int n, const double* x);
extern __DPCPP_SYCL_EXTERNAL double __imf_norm3d (double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_norm4d (double x, double y, double z, double w);
extern __DPCPP_SYCL_EXTERNAL double __imf_pow (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_pow_ep (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_pow_ha (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_pow_la (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_pown (double x, int y);
extern __DPCPP_SYCL_EXTERNAL double __imf_rcbrt (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_remainder (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_remquo (double x, double y, int* q);
extern __DPCPP_SYCL_EXTERNAL double __imf_rhypot (double x, double y);
extern __DPCPP_SYCL_EXTERNAL double __imf_rnorm (int n, const double* x);
extern __DPCPP_SYCL_EXTERNAL double __imf_rnorm3d (double x, double y, double z);
extern __DPCPP_SYCL_EXTERNAL double __imf_rnorm4d (double x, double y, double z, double w);
extern __DPCPP_SYCL_EXTERNAL double __imf_round (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_scalbn (double x, int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_signbit (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sin (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sin_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sin_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sin_la (double x);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincos (double x, double* s, double* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincos_ep (double x, double* s, double* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincos_ha (double x, double* s, double* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincos_la (double x, double* s, double* c);
extern __DPCPP_SYCL_EXTERNAL void __imf_sincospi (double x, double* s, double* c);
extern __DPCPP_SYCL_EXTERNAL double __imf_sinh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sinh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sinh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sinh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_sinpi (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tan (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tan_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tan_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tan_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tanh (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tanh_ep (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tanh_ha (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tanh_la (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_tgamma (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_y0 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_y1 (double x);
extern __DPCPP_SYCL_EXTERNAL double __imf_yn (int n, double x);
/*
 * IMF math builtins end
 */
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmax_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmax_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmax_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmin_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_viaddmin_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_viaddmin_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmax_s16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL int __imf_vibmax_s32(int x, int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmax_u16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vibmax_u32(unsigned int x, unsigned int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmin_s16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL int __imf_vibmin_s32(int x, int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int __imf_vibmin_u16x2(unsigned int x,
                                                             unsigned int y,
                                                             bool *pred_hi,
                                                             bool *pred_lo);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vibmin_u32(unsigned int x, unsigned int y, bool *pred);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_s16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_s16x2_relu(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax3_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax3_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin3_s32(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin3_s32_relu(int x, int y, int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax3_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_u16x2(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin3_u32(unsigned int x, unsigned int y, unsigned int z);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimax_s16x2_relu(unsigned int x, unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimax_s32_relu(int x, int y);
extern __DPCPP_SYCL_EXTERNAL unsigned int
__imf_vimin_s16x2_relu(unsigned int x, unsigned int y);
extern __DPCPP_SYCL_EXTERNAL int __imf_vimin_s32_relu(int x, int y);
}
#ifdef __GLIBC__
extern "C" {
extern __DPCPP_SYCL_EXTERNAL void __assert_fail(const char *expr,
                                                const char *file,
                                                unsigned int line,
                                                const char *func);
}
#elif defined(_WIN32)
extern "C" {
// TODO: documented C runtime library APIs must be recognized as
//       builtins by FE. This includes _dpcomp, _dsign, _dtest,
//       _fdpcomp, _fdsign, _fdtest, _hypotf, _wassert.
//       APIs used by STL, such as _Cosh, are undocumented, even though
//       they are open-sourced. Recognizing them as builtins is not
//       straightforward currently.
extern __DPCPP_SYCL_EXTERNAL void _wassert(const wchar_t *wexpr,
                                           const wchar_t *wfile, unsigned line);
}
#endif
#endif // __SYCL_DEVICE_ONLY__
