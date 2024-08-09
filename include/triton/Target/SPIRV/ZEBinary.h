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
  printf("Compiling ze native code\n");
  auto binary_ptr = const_cast<uint8_t *>(
      reinterpret_cast<const uint8_t *>(spirv_kernel.data()));

  sycl::device *sycl_device_ptr =
      reinterpret_cast<sycl::device *>(sycl_device_addr);
  sycl::device sycl_device = *sycl_device_ptr;
  printf("Casted device ptr\nb");
  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  printf("Accessed sycl device ptr\n");
  auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  // compile the spirv module
  printf("Creating module of size %ld...\n", spirv_kernel.size());
  printf("kernel ptr: %p\n", binary_ptr);
  printf("kernel str ptr: %p\n", spirv_kernel.data());

  // TODO: bring GRF switch over
  std::string build_flags = build_flags_in + "  -cl-intel-256-GRF-per-thread";
  auto l0_module =
      checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                    spirv_kernel.size(), build_flags.c_str()));
  printf("Created module\n");

  // TODO: do we care about the kernel?
  auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));
  printf("Created function.");
  size_t module_size_bytes;
  gpuAssert(zeModuleGetNativeBinary(l0_module, &module_size_bytes, nullptr));
  std::cout << "Serializing " << module_size_bytes << " byte module to disk..."
            << std::endl;
  std::string serialized_module;
  serialized_module.resize(module_size_bytes);
  gpuAssert(zeModuleGetNativeBinary(
      l0_module, &module_size_bytes,
      reinterpret_cast<uint8_t *>(serialized_module.data())));

  return serialized_module;
}

#endif
