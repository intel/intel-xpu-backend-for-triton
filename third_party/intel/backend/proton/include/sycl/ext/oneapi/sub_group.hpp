//
// Modifications, Copyright (C) 2021 Intel Corporation
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
//==----------- sub_group.hpp --- SYCL sub-group ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_DEPRECATED
#include <sycl/sub_group.hpp>                 // for sub_group

#include <tuple> // for _Swallow_assign, ignore

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
struct __SYCL_DEPRECATED("use sycl::sub_group() instead") sub_group
    : sycl::sub_group {
  // These two constructors are intended to keep the correctness of such code
  // after the sub_group class migration from ext::oneapi to the sycl namespace:
  // sycl::ext::oneapi::sub_group sg =
  //    sycl::ext::oneapi::experimental::this_sub_group();
  // ...
  // sycl::ext::oneapi::sub_group sg = item.get_sub_group();
  // Note: this constructor is used for implicit conversion. Since the
  // sub_group class doesn't have any members, just ignore the arg.
  sub_group(const sycl::sub_group &sg) : sub_group() { std::ignore = sg; }

private:
  sub_group() = default;
};
} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
