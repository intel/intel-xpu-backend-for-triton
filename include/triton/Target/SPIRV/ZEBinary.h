#ifndef TRITON_TARGET_ZEBINARY_H
#define TRITON_TARGET_ZEBINARY_H

#include "third_party/intel/backend/include/sycl_functions.h"

static inline void gpuAssert(ze_result_t code) {
  if (code != ZE_RESULT_SUCCESS) {
    auto str = parseZeResultCode(code);
    throw std::runtime_error(str);
  }
}

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  gpuAssert(std::get<1>(tuple));
  return std::get<0>(tuple);
}

std::string compile_ze_native_code(const std::string &kernel_name,
                                   const std::string &build_flags_in,
                                   int shared, uint64_t sycl_device_addr,
                                   const std::string &spirv_kernel) {
  auto binary_ptr = const_cast<uint8_t *>(
      reinterpret_cast<const uint8_t *>(spirv_kernel.data()));

  // initialize sycl compilation runtime
  // TODO: might be nice to initialize once and cache it - is this code re-entrant? 
  sycl::default_selector d_selector;

  sycl::queue q(d_selector, exception_handler);

  auto sycl_context = q.get_context();
  std::vector<sycl::device> sycl_devices = sycl_context.get_devices();
  assert(sycl_devices.size() > 0);
  auto sycl_device = sycl_devices.front();
  auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  auto l0_module =
      checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                    spirv_kernel.size(), build_flags_in.c_str()));

  // TODO: do we care about the kernel? - yes if we port the GRF code 
  // auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));
  size_t module_size_bytes;
  gpuAssert(zeModuleGetNativeBinary(l0_module, &module_size_bytes, nullptr));
  std::string serialized_module;
  serialized_module.resize(module_size_bytes);
  gpuAssert(zeModuleGetNativeBinary(
      l0_module, &module_size_bytes,
      reinterpret_cast<uint8_t *>(serialized_module.data())));

  return serialized_module;
}

#endif
