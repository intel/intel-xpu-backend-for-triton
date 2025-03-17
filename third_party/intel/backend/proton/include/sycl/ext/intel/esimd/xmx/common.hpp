//
// Modifications, Copyright (C) 2022 Intel Corporation
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
//==-------------- xmx/common.hpp - DPC++ Explicit SIMD API ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Explicit SIMD API types used in ESIMD
// Intel(R) Xe Matrix eXtensions (Intel(R) XMX).
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::xmx {

/// Describes the element types in the input matrices.
/// Used as template parameter to dpas() and may be omitted when
/// it is deducible from the element types of input matrices.
enum class dpas_argument_type {
  Invalid = 0,
  u1 __SYCL_DEPRECATED("u1 is reserved/unsupported") = 1, // unsigned 1 bit
  s1 __SYCL_DEPRECATED("s1 is reserved/unsupported") = 2, // signed 1 bit
  u2 = 3,                                                 // unsigned 2 bits
  s2 = 4,                                                 // signed 2 bits
  u4 = 5,                                                 // unsigned 4 bits
  s4 = 6,                                                 // signed 4 bits
  u8 = 7,                                                 // unsigned 8 bits
  s8 = 8,                                                 // signed 8 bits
  bf16 = 9,                                               // bfloat 16
  fp16 = 10,                                              // half float
  tf32 = 12, // tensorfloat 32
};

} // namespace ext::intel::esimd::xmx
} // namespace _V1
} // namespace sycl
