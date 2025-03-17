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
//==------------ memory_intrin.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares experimental memory Explicit SIMD intrinsics.
//===----------------------------------------------------------------------===//

/// @cond ESIMD_DETAIL

#pragma once

#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>


// generic work-group split barrier
__ESIMD_INTRIN void
__esimd_sbarrier(__ESIMD_ENS::split_barrier_action flag) __ESIMD_INTRIN_END;

#ifdef __SYCL_DEVICE_ONLY__
// Create an explicit data and GPU scoreboard dependency.
__ESIMD_INTRIN void __esimd_wait(uint16_t value);
#endif // __SYCL_DEVICE_ONLY__

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
template <__ESIMD_NS::memory_kind Kind, __ESIMD_NS::fence_flush_op FenceOp,
          __ESIMD_NS::fence_scope Scope, int N>
__ESIMD_INTRIN void
__esimd_lsc_fence(__ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

__ESIMD_INTRIN uint8_t __esimd_named_barrier_allocate(uint8_t NbarCount)
    __ESIMD_INTRIN_END;

/// @endcond ESIMD_DETAIL
