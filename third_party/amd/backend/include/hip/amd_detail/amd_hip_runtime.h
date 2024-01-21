/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 *  @file  amd_detail/hip_runtime.h
 *  @brief Contains definitions of APIs for HIP runtime.
 */

//#pragma once
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_H

#include <hip/amd_detail/amd_hip_common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Query the installed library build name.
 *
 * This function can be used even when the library is not initialized.
 *
 * @returns Returns a string describing the build version of the library.  The
 * string is owned by the library.
 */
const char* amd_dbgapi_get_build_name();

/**
 * @brief Query the installed library git hash.
 *
 * This function can be used even when the library is not initialized.
 *
 * @returns Returns git hash of the library.
 */
const char* amd_dbgapi_get_git_hash();

/**
 * @brief Query the installed library build ID.
 *
 * This function can be used even when the library is not initialized.
 *
 * @returns Returns build ID of the library.
 */
size_t amd_dbgapi_get_build_id();

#ifdef __cplusplus
} /* extern "c" */
#endif

//---
// Top part of file can be compiled with any compiler

#if !defined(__HIPCC_RTC__)
//#include <cstring>
#if __cplusplus
#include <cmath>
#include <cstdint>
#else
#include <math.h>
#include <string.h>
#include <stddef.h>
#endif // __cplusplus
#else
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef signed long long int64_t;
namespace std {
using ::uint32_t;
using ::uint64_t;
using ::int32_t;
using ::int64_t;
}
#endif // !defined(__HIPCC_RTC__)

#if __HIP_CLANG_ONLY__

#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#define CUDA_SUCCESS hipSuccess

#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime_api.h>
extern int HIP_TRACE_API;
#endif // !defined(__HIPCC_RTC__)

#ifdef __cplusplus
#include <hip/amd_detail/hip_ldg.h>
#endif
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/host_defines.h>
#include <hip/amd_detail/amd_device_functions.h>
#include <hip/amd_detail/amd_surface_functions.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/texture_indirect_functions.h>

// TODO-HCC remove old definitions ; ~1602 hcc supports __HCC_ACCELERATOR__ define.
#if defined(__KALMAR_ACCELERATOR__) && !defined(__HCC_ACCELERATOR__)
#define __HCC_ACCELERATOR__ __KALMAR_ACCELERATOR__
#endif

// Feature tests:
#if (defined(__HCC_ACCELERATOR__) && (__HCC_ACCELERATOR__ != 0)) || __HIP_DEVICE_COMPILE__
// Device compile and not host compile:

// 32-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (1)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (1)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (1)

// 64-bit Atomics:
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (1)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (1)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (1)

// warp cross-lane operations:
#define __HIP_ARCH_HAS_WARP_VOTE__ (1)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (1)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (1)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (1)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (1)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)

#endif /* Device feature flags */


#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                            \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)                \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                     \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...)                                                                     \
  select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0, )(__VA_ARGS__)

#if !defined(__HIPCC_RTC__)
__host__ inline void* __get_dynamicgroupbaseptr() { return nullptr; }
#endif // !defined(__HIPCC_RTC__)

// End doxygen API:
/**
 *   @}
 */

//
// hip-clang functions
//
#if !defined(__HIPCC_RTC__)
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) X

typedef int hipLaunchParm;

template <std::size_t n, typename... Ts,
          typename std::enable_if<n == sizeof...(Ts)>::type* = nullptr>
void pArgs(const std::tuple<Ts...>&, void*) {}

template <std::size_t n, typename... Ts,
          typename std::enable_if<n != sizeof...(Ts)>::type* = nullptr>
void pArgs(const std::tuple<Ts...>& formals, void** _vargs) {
    using T = typename std::tuple_element<n, std::tuple<Ts...> >::type;

    static_assert(!std::is_reference<T>{},
                  "A __global__ function cannot have a reference as one of its "
                  "arguments.");
#if defined(HIP_STRICT)
    static_assert(std::is_trivially_copyable<T>{},
                  "Only TriviallyCopyable types can be arguments to a __global__ "
                  "function");
#endif
    _vargs[n] = const_cast<void*>(reinterpret_cast<const void*>(&std::get<n>(formals)));
    return pArgs<n + 1>(formals, _vargs);
}

template <typename... Formals, typename... Actuals>
std::tuple<Formals...> validateArgsCountType(void (*kernel)(Formals...), std::tuple<Actuals...>(actuals)) {
    static_assert(sizeof...(Formals) == sizeof...(Actuals), "Argument Count Mismatch");
    std::tuple<Formals...> to_formals{std::move(actuals)};
    return to_formals;
}

#if defined(HIP_TEMPLATE_KERNEL_LAUNCH)
template <typename... Args, typename F = void (*)(Args...)>
void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                        std::uint32_t sharedMemBytes, hipStream_t stream, Args... args) {
    constexpr size_t count = sizeof...(Args);
    auto tup_ = std::tuple<Args...>{args...};
    auto tup = validateArgsCountType(kernel, tup_);
    void* _Args[count];
    pArgs<0>(tup, _Args);

    auto k = reinterpret_cast<void*>(kernel);
    hipLaunchKernel(k, numBlocks, dimBlocks, _Args, sharedMemBytes, stream);
}
#else
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads, memPerBlock, streamId, ...)  \
    do {                                                                                           \
        kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>(__VA_ARGS__);         \
    } while (0)

#define hipLaunchKernelGGL(kernelName, ...)  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)
#endif

#include <hip/hip_runtime_api.h>
#endif // !defined(__HIPCC_RTC__)

extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_id(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_group_id(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_size(uint);
extern "C" __device__ __attribute__((const)) size_t __ockl_get_num_groups(uint);
struct __HIP_BlockIdx {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept { return __ockl_get_group_id(x); }
};
struct __HIP_BlockDim {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_size(x);
  }
};
struct __HIP_GridDim {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_num_groups(x);
  }
};
struct __HIP_ThreadIdx {
  __device__
  std::uint32_t operator()(std::uint32_t x) const noexcept {
    return __ockl_get_local_id(x);
  }
};

#if defined(__HIPCC_RTC__)
typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
#ifdef __cplusplus
    constexpr __device__ dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) : x(_x), y(_y), z(_z){};
#endif
} dim3;
#endif // !defined(__HIPCC_RTC__)

template <typename F>
struct __HIP_Coordinates {
  using R = decltype(F{}(0));

  struct __X {
    __device__ operator R() const noexcept { return F{}(0); }
    __device__ R operator+=(const R& rhs) { return F{}(0) + rhs; }
  };
  struct __Y {
    __device__ operator R() const noexcept { return F{}(1); }
    __device__ R operator+=(const R& rhs) { return F{}(1) + rhs; }
  };
  struct __Z {
    __device__ operator R() const noexcept { return F{}(2); }
    __device__ R operator+=(const R& rhs) { return F{}(2) + rhs; }
  };

  static constexpr __X x{};
  static constexpr __Y y{};
  static constexpr __Z z{};
#ifdef __cplusplus
  __device__ operator dim3() const { return dim3(x, y, z); }
#endif

};
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::__X __HIP_Coordinates<F>::x;
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::__Y __HIP_Coordinates<F>::y;
template <typename F>
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
constexpr typename __HIP_Coordinates<F>::__Z __HIP_Coordinates<F>::z;

extern "C" __device__ __attribute__((const)) size_t __ockl_get_global_size(uint);
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__X,
                        __HIP_Coordinates<__HIP_BlockDim>::__X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__X,
                        __HIP_Coordinates<__HIP_GridDim>::__X) noexcept {
  return __ockl_get_global_size(0);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__Y,
                        __HIP_Coordinates<__HIP_BlockDim>::__Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__Y,
                        __HIP_Coordinates<__HIP_GridDim>::__Y) noexcept {
  return __ockl_get_global_size(1);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_GridDim>::__Z,
                        __HIP_Coordinates<__HIP_BlockDim>::__Z) noexcept {
  return __ockl_get_global_size(2);
}
inline
__device__
std::uint32_t operator*(__HIP_Coordinates<__HIP_BlockDim>::__Z,
                        __HIP_Coordinates<__HIP_GridDim>::__Z) noexcept {
  return __ockl_get_global_size(2);
}

static constexpr __HIP_Coordinates<__HIP_BlockDim> blockDim{};
static constexpr __HIP_Coordinates<__HIP_BlockIdx> blockIdx{};
static constexpr __HIP_Coordinates<__HIP_GridDim> gridDim{};
static constexpr __HIP_Coordinates<__HIP_ThreadIdx> threadIdx{};

extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_id(uint);
#define hipThreadIdx_x (__ockl_get_local_id(0))
#define hipThreadIdx_y (__ockl_get_local_id(1))
#define hipThreadIdx_z (__ockl_get_local_id(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_group_id(uint);
#define hipBlockIdx_x (__ockl_get_group_id(0))
#define hipBlockIdx_y (__ockl_get_group_id(1))
#define hipBlockIdx_z (__ockl_get_group_id(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_local_size(uint);
#define hipBlockDim_x (__ockl_get_local_size(0))
#define hipBlockDim_y (__ockl_get_local_size(1))
#define hipBlockDim_z (__ockl_get_local_size(2))

extern "C" __device__ __attribute__((const)) size_t __ockl_get_num_groups(uint);
#define hipGridDim_x (__ockl_get_num_groups(0))
#define hipGridDim_y (__ockl_get_num_groups(1))
#define hipGridDim_z (__ockl_get_num_groups(2))

#include <hip/amd_detail/amd_math_functions.h>

#if __HIP_HCC_COMPAT_MODE__
// Define HCC work item functions in terms of HIP builtin variables.
#pragma push_macro("__DEFINE_HCC_FUNC")
#define __DEFINE_HCC_FUNC(hc_fun,hip_var) \
inline __device__ __attribute__((always_inline)) uint hc_get_##hc_fun(uint i) { \
  if (i==0) \
    return hip_var.x; \
  else if(i==1) \
    return hip_var.y; \
  else \
    return hip_var.z; \
}

__DEFINE_HCC_FUNC(workitem_id, threadIdx)
__DEFINE_HCC_FUNC(group_id, blockIdx)
__DEFINE_HCC_FUNC(group_size, blockDim)
__DEFINE_HCC_FUNC(num_groups, gridDim)
#pragma pop_macro("__DEFINE_HCC_FUNC")

extern "C" __device__ __attribute__((const)) size_t __ockl_get_global_id(uint);
inline __device__ __attribute__((always_inline)) uint
hc_get_workitem_absolute_id(int dim)
{
  return (uint)__ockl_get_global_id(dim);
}

#endif

#if !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#if !defined(__HIPCC_RTC__)
// Support std::complex.
#if !_OPENMP || __HIP_ENABLE_CUDA_WRAPPER_FOR_OPENMP__
#pragma push_macro("__CUDA__")
#define __CUDA__
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
// Workaround for using libc++ with HIP-Clang.
// The following headers requires clang include path before standard C++ include path.
// However libc++ include path requires to be before clang include path.
// To workaround this, we pass -isystem with the parent directory of clang include
// path instead of the clang include path itself.
#include <include/cuda_wrappers/algorithm>
#include <include/cuda_wrappers/complex>
#include <include/cuda_wrappers/new>
#undef __CUDA__
#pragma pop_macro("__CUDA__")
#endif // !_OPENMP || __HIP_ENABLE_CUDA_WRAPPER_FOR_OPENMP__
#endif // !defined(__HIPCC_RTC__)
#endif // !__CLANG_HIP_RUNTIME_WRAPPER_INCLUDED__
#endif // __HIP_CLANG_ONLY__

#endif  // HIP_AMD_DETAIL_RUNTIME_H
