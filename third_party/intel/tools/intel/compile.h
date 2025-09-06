/* clang-format off */
#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <inttypes.h>
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <sycl/sycl.hpp>

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(sycl::queue &stream);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
int32_t{_placeholder} {kernel_name}(sycl::queue &stream, {signature});
