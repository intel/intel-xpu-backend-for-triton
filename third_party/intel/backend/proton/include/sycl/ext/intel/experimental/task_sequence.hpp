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
//==--------------------------- task_sequence.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>
#include <sycl/aspects.hpp>
#include <sycl/exception.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace intel {
namespace experimental {

template <auto &f, uint32_t invocation_capacity = 1,
          uint32_t response_capacity = 1>
struct task_sequence;

template <typename ReturnT, typename... ArgsT, ReturnT (&f)(ArgsT...),
          uint32_t invocation_capacity, uint32_t response_capacity>
class task_sequence<f, invocation_capacity, response_capacity> {
  // TODO: put atomic lock on it if required
  unsigned outstanding = 0;
  size_t id;

  typedef ReturnT (*f_t)(ArgsT...);

public:
  task_sequence(const task_sequence &) = delete;
  task_sequence &operator=(const task_sequence &) = delete;
  task_sequence(task_sequence &&) = delete;
  task_sequence &operator=(task_sequence &&) = delete;

  task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    // Currently it is decided to use some useless value for 3rd argument to not
    // impact code in other components. API will be aligned with the spec later.
    // TODO: align API with the feature specs.
    id = __spirv_TaskSequenceCreateINTEL(this, &f, 0);

#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  void async(ArgsT... Args) {
#if defined(__SYCL_DEVICE_ONLY__)
    ++outstanding;
    __spirv_TaskSequenceAsyncINTEL(this, &f, id, invocation_capacity, Args...);
#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  ReturnT get() {
#if defined(__SYCL_DEVICE_ONLY__)
    --outstanding;
    return __spirv_TaskSequenceGetINTEL(this, &f, id, response_capacity);
#else
    throw exception{errc::feature_not_supported,
                    "task_sequence is not supported on host device"};
#endif
  }

  ~task_sequence() {
#if defined(__SYCL_DEVICE_ONLY__)
    while (outstanding)
      get();
    __spirv_TaskSequenceReleaseINTEL(this);
#else
    // "task_sequence is not supported on host device"
    // Destructor shouldn't throw exception.
#endif
  }
};

} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace _V1
} // namespace sycl
