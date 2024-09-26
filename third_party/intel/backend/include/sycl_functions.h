#ifndef SYCL_FUNCTIONS_INCLUDE_H_
#define SYCL_FUNCTIONS_INCLUDE_H_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

typedef struct l0_resc_handles {
  ze_context_handle_t context;
  ze_device_handle_t device;
  ze_command_queue_handle_t queue;
  ze_command_list_handle_t cmd_list;
} l0_resc_handles;

using SyclQueueMap = std::unordered_map<sycl::queue, l0_resc_handles>;

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

inline std::string parseZeResultCode(const ze_result_t code) {
  const std::string prefix = "Triton Error [ZE]: ";
  std::stringstream ss;
  ss << prefix << "0x" << std::hex << code << "\n";
  return ss.str();
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
  auto error_no =
      zeModuleCreate(context, device, &module_description, &module, &buildlog);
  if (error_no != ZE_RESULT_SUCCESS) {
    size_t szLog = 0;
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, nullptr));
    char *strLog = (char *)malloc(szLog);
    ZE_CHECK(zeModuleBuildLogGetString(buildlog, &szLog, strLog));
    std::cerr << "L0 build module failed. Log: " << strLog << std::endl;
    free(strLog);
    ZE_CHECK(zeModuleBuildLogDestroy(buildlog));
  }
  ZE_CHECK(error_no);
  return std::make_tuple(module, error_no);
}

std::tuple<ze_kernel_handle_t, ze_result_t>
create_function(ze_module_handle_t module, ze_kernel_flags_t flag,
                std::string_view func_name) {
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
  ZE_CHECK(zeKernelCreate(module, &kernel_description, &kernel));
  return std::make_tuple(kernel, ZE_RESULT_SUCCESS);
}

std::tuple<ze_kernel_handle_t, ze_result_t>
create_function(ze_module_handle_t module, std::string_view func_name) {
  return create_function(module, ZE_KERNEL_FLAG_FORCE_RESIDENCY, func_name);
}

void printModuleKernelName(ze_module_handle_t hModule) {
  uint32_t Count = 0;
  auto ret = zeModuleGetKernelNames(hModule, &Count, nullptr);
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

std::vector<sycl::device> update(sycl::queue sycl_queue,
                                 SyclQueueMap &sycl_queue_map) {
  // Get l0-context
  auto sycl_context = sycl_queue.get_context();
  ze_context_handle_t hCtxt =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_context);
  // Get l0-device
  const std::vector<sycl::device> &sycl_devices = sycl_context.get_devices();
  assert(sycl_devices.size() > 0);
  ze_device_handle_t hDev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          sycl_devices.front());
  // Get l0-queue
  bool immediate_cmd_list = false;
  std::variant<ze_command_queue_handle_t, ze_command_list_handle_t> queue_var =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_queue);
  auto l0_queue = std::get_if<ze_command_queue_handle_t>(&queue_var);
  if (l0_queue == nullptr) {
    auto imm_cmd_list = std::get_if<ze_command_list_handle_t>(&queue_var);
    if (imm_cmd_list == nullptr) {
      return {};
    }
    immediate_cmd_list = true;
    sycl_queue_map[sycl_queue].cmd_list = *imm_cmd_list;
  }
  sycl_queue_map[sycl_queue].context = hCtxt;
  sycl_queue_map[sycl_queue].device = hDev;
  sycl_queue_map[sycl_queue].queue = immediate_cmd_list ? 0 : *l0_queue;

  return sycl_devices;
}

#endif // SYCL_FUNCTIONS_INCLUDE_H_
