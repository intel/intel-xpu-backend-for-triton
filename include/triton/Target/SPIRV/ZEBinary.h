#ifndef TRITON_TARGET_ZEBINARY_H
#define TRITON_TARGET_ZEBINARY_H

#include "third_party/intel/backend/include/sycl_functions.h"

#include "third_party/intel/backend/include/measure.h"

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
  // TODO: might be nice to initialize once and cache it - is this code
  // re-entrant?
  sycl::default_selector d_selector;

  sycl::queue q(d_selector, exception_handler);

  auto sycl_context = q.get_context();
  std::vector<sycl::device> sycl_devices = sycl_context.get_devices();
  assert(sycl_devices.size() > 0);
  auto sycl_device = sycl_devices.front();
  auto l0_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);

  auto ctx = sycl_device.get_platform().ext_oneapi_get_default_context();
  auto l0_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  ze_module_handle_t l0_module;
  auto create_module_ms = measure<>::execution([&]() {
    l0_module = checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                              spirv_kernel.size(),
                                              build_flags_in.c_str()));
  });
  std::cout << "Module creation time: " << create_module_ms << " ms"
            << std::endl;
  if (!l0_module) {
    throw std::runtime_error("Failed to create L0 module!");
  }

  // TODO: module create logging

  auto checkL0Errors = [&](auto l0_module) -> ze_kernel_handle_t {
    ze_kernel_handle_t l0_kernel;
    auto create_function_ms = measure<>::execution([&]() {
      l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));
    });
    std::cout << "Function creation time: " << create_function_ms << " ms"
              << std::endl;
    if (!l0_kernel) {
      throw std::runtime_error("Failed to create L0 Kernel!");
    }
    return l0_kernel;
  };
  // Retrieve the kernel properties (e.g. register spills).
  ze_kernel_handle_t l0_kernel = checkL0Errors(l0_module);
  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  gpuAssert(zeKernelGetProperties(l0_kernel, &props));

  int32_t n_spills = props.spillMemSize;
  int32_t n_regs = 0;
  constexpr int32_t max_reg_spill = 1000;
  std::string build_flags(build_flags_in);
  bool is_GRF_mode_specified = false;
  // Check whether the GRF mode is specified by the build flags.
  if (build_flags.find("-cl-intel-256-GRF-per-thread") != std::string::npos ||
      build_flags.find("-cl-intel-128-GRF-per-thread") != std::string::npos ||
      build_flags.find("-cl-intel-enable-auto-large-GRF-mode") !=
          std::string::npos) {
    is_GRF_mode_specified = true;
  }
  if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
    std::cout << "(I): Detected " << n_spills
              << " spills, recompiling the kernel using large GRF mode"
              << std::endl;
    const std::string new_build_flags =
        build_flags.append(" -cl-intel-256-GRF-per-thread");
    auto create_module_ms = measure<>::execution([&]() {
      l0_module = checkSyclErrors(create_module(l0_context, l0_device,
                                                binary_ptr, spirv_kernel.size(),
                                                new_build_flags.c_str()));
    });
    std::cout << "Large GRF Module creation time: " << create_module_ms << " ms"
              << std::endl;
    if (!l0_module) {
      throw std::runtime_error("Failed to create L0 module!");
    }

    // TODO: module create logging
    l0_kernel = checkL0Errors(l0_module);
    gpuAssert(zeKernelGetProperties(l0_kernel, &props));
    n_spills = props.spillMemSize;
    std::cout << "(I): Kernel has now " << n_spills << " spills" << std::endl;
  }

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
