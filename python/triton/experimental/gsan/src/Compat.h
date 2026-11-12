#pragma once

#if defined(__CUDA__) && defined(__clang__)
#define GSAN_DEVICE __attribute__((device))
#define GSAN_HOST_DEVICE __attribute__((host)) __attribute__((device))
#elif defined(__CUDACC__)
#define GSAN_DEVICE __device__
#define GSAN_HOST_DEVICE __host__ __device__
#else
#define GSAN_DEVICE
#define GSAN_HOST_DEVICE
#endif

namespace gsan {

#ifdef _MSC_VER
// MSVC does not provide GCC/Clang __SIZE_TYPE__ builtins.
// Use MSVC built-in sized types directly (no headers needed).
using uint8_t = unsigned __int8;
using uint16_t = unsigned __int16;
using uint32_t = unsigned __int32;
using uint64_t = unsigned __int64;
#ifdef _WIN64
using size_t = unsigned __int64;
using uintptr_t = unsigned __int64;
#else
using size_t = unsigned __int32;
using uintptr_t = unsigned __int32;
#endif
#else
using size_t = __SIZE_TYPE__;
using int16_t = __INT16_TYPE__;
using int64_t = __INT64_TYPE__;
using uint8_t = __UINT8_TYPE__;
using uint16_t = __UINT16_TYPE__;
using uint32_t = __UINT32_TYPE__;
using uintptr_t = __UINTPTR_TYPE__;
using uint64_t = __UINT64_TYPE__;
#endif

} // namespace gsan
