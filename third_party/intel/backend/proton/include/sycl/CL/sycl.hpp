//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#warning "CL/sycl.hpp is deprecated, use sycl/sycl.hpp"
#pragma clang diagnostic pop
#endif

#include <sycl/sycl.hpp>

namespace cl {
namespace sycl = ::sycl;
}
