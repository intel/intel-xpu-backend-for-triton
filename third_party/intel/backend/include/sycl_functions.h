#ifndef SYCL_FUNCTIONS_INCLUDE_H_
#define SYCL_FUNCTIONS_INCLUDE_H_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

// Create an exception handler for asynchronous SYCL exceptions
auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

inline std::string parseZeResultCode(const ze_result_t result) {
  // https://github.com/oneapi-src/level-zero/blob/master/source/utils/logging.cpp#L12
  if (result == ZE_RESULT_SUCCESS) {
    return "ZE_RESULT_SUCCESS";
  } else if (result == ZE_RESULT_NOT_READY) {
    return "ZE_RESULT_NOT_READY";
  } else if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  } else if (result == ZE_RESULT_ERROR_DEVICE_LOST) {
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  } else if (result == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  } else if (result == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  } else if (result == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  } else if (result == ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS) {
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  } else if (result == ZE_RESULT_ERROR_NOT_AVAILABLE) {
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  } else if (result == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE) {
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  } else if (result == ZE_RESULT_WARNING_DROPPED_DATA) {
    return "ZE_RESULT_WARNING_DROPPED_DATA";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_VERSION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_HANDLE) {
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  } else if (result == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE) {
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_POINTER) {
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  } else if (result == ZE_RESULT_ERROR_INVALID_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  } else if (result == ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT) {
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  } else if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  } else if (result == ZE_RESULT_ERROR_INVALID_NATIVE_BINARY) {
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  } else if (result == ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED) {
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  } else if (result == ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE) {
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  } else if (result == ZE_RESULT_ERROR_OVERLAPPING_REGIONS) {
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  } else if (result == ZE_RESULT_ERROR_UNKNOWN) {
    return "ZE_RESULT_ERROR_UNKNOWN";
  } else {
    return std::to_string(static_cast<int>(result));
  }
}

#define ZE_CHECK(code)                                                         \
  {                                                                            \
    if (code != ZE_RESULT_SUCCESS) {                                           \
      return std::make_tuple(nullptr, code);                                   \
    }                                                                          \
  }

// TODO: share Triton GetEnv.hpp impl
inline std::string getStrEnv(const std::string &env) {
  const char *cstr = std::getenv(env.c_str());
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

inline std::optional<bool> isEnvValueBool(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (str == "on" || str == "true" || str == "1")
    return true;
  if (str == "off" || str == "false" || str == "0")
    return false;
  return std::nullopt;
}

std::tuple<ze_module_handle_t, ze_result_t>
create_module(ze_context_handle_t context, ze_device_handle_t device,
              uint8_t *binary_ptr, size_t binary_size, const char *build_flags,
              const bool is_spv = true) {
  std::cout << "# [mdziado][create_mobule][1] called" << std::endl;
  assert(binary_ptr != nullptr && "binary_ptr should not be NULL");
  assert(build_flags != nullptr && "build_flags should not be NULL");

  const ze_module_format_t format =
      is_spv ? ZE_MODULE_FORMAT_IL_SPIRV : ZE_MODULE_FORMAT_NATIVE;
  ze_module_desc_t module_description = {};
  module_description.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  module_description.format = format;
  module_description.inputSize = static_cast<uint32_t>(binary_size);
  module_description.pInputModule = binary_ptr;
  module_description.pBuildFlags = build_flags;
  ze_module_build_log_handle_t buildlog;
  ze_module_handle_t module;
  std::cout << "# [mdziado][create_mobule][2] before zeModuleCreate" << std::endl;

  auto error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);
  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
    char *strLog = (char *)malloc(szLog);
    auto error_no_build_log =
        zeModuleBuildLogGetString(buildlog, &szLog, strLog);
    if (error_no_build_log != ZE_RESULT_SUCCESS) {
      free(strLog);
      ZE_CHECK(error_no_build_log);
    }
    std::cout << "L0 build module failed. Log: " << strLog << std::endl;
    free(strLog);
    ZE_CHECK(zeModuleBuildLogDestroy(buildlog));
  }
  ZE_CHECK(error_no);

  std::cout << "# [mdziado][create_mobule][1] before return" << std::endl;

  return std::make_tuple(module, error_no);
}

std::tuple<ze_kernel_handle_t, ze_result_t>
create_function(ze_module_handle_t module, ze_kernel_flags_t flag,
                std::string_view func_name) {
  std::cout << "# [mdziado][create_function][1] called" << std::endl;

  ze_kernel_handle_t kernel;
  ze_kernel_desc_t kernel_description = {};
  kernel_description.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernel_description.pNext = nullptr;
  kernel_description.flags = flag;
  kernel_description.pKernelName = func_name.data();
  assert(module);
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    std::cout << "create kernel:" << func_name << std::endl;
  }

  uint32_t count = 0;
  std::cout << "# [mdziado][create_function][2] before zeModuleGetKernelNames" << std::endl;

  [[maybe_unused]] auto ret = zeModuleGetKernelNames(module, &count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  if (count == 0) {
    std::cout
        << "Module doesn't contain any kernels. Please attempt to tune the "
           "'num_warps' parameter."
        << std::endl;
    return std::make_tuple(nullptr, ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
  }

  std::cout << "# [mdziado][create_function][3] before zeKernelCreate" << std::endl;
  ZE_CHECK(zeKernelCreate(module, &kernel_description, &kernel));
  std::cout << "# [mdziado][create_function][4] before return" << std::endl;

  return std::make_tuple(kernel, ZE_RESULT_SUCCESS);
}

std::tuple<ze_kernel_handle_t, ze_result_t>
create_function(ze_module_handle_t module, std::string_view func_name) {
  return create_function(module, ZE_KERNEL_FLAG_FORCE_RESIDENCY, func_name);
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  [[maybe_unused]] auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
  assert(ret == ZE_RESULT_SUCCESS);
  std::unique_ptr<const char *[]> PNames(new const char *[Count]);
  ret = zeModuleGetKernelNames(hModule, &Count, PNames.get());
  assert(ret == ZE_RESULT_SUCCESS);
  if (getBoolEnv("MLIR_ENABLE_DUMP")) {
    for (uint32_t i = 0; i < Count; ++i) {
      std::cout << std::string(PNames[i]) << std::endl;
    }
  }
}

#endif // SYCL_FUNCTIONS_INCLUDE_H_
