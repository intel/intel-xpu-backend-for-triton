/* clang-format off */
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <inttypes.h>
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <sycl/sycl.hpp>

#endif

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC
#endif

EXPORT_FUNC void unload_{kernel_name}(void);
EXPORT_FUNC void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
EXPORT_FUNC int32_t{_placeholder} {kernel_name}(sycl::queue &stream, {signature});
