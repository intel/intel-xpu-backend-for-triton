//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

#include "sycl_functions.h"

#include <Python.h>

static std::vector<std::pair<sycl::device, ze_device_handle_t>>
    g_sycl_l0_device_list;

static std::vector<sycl::device> sycl_opencl_device_list;

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  const auto code = std::get<1>(tuple);
  if (code != ZE_RESULT_SUCCESS) {
    throw std::runtime_error(parseZeResultCode(code));
  }
  return std::get<0>(tuple);
}

extern "C" EXPORT_FUNC PyObject *get_device_properties(int device_id) {
  if (device_id > g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }
  const auto device = g_sycl_l0_device_list[device_id];

  // Get device handle
  ze_device_handle_t phDevice = device.second;

  // create a struct to hold device properties
  ze_device_properties_t device_properties = {};
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  zeDeviceGetProperties(phDevice, &device_properties);

  int multiprocessor_count =
      device_properties.numSlices * device_properties.numSubslicesPerSlice;
  int sm_clock_rate = device_properties.coreClockRate;

  ze_device_compute_properties_t compute_properties = {};
  compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  zeDeviceGetComputeProperties(phDevice, &compute_properties);
  int max_shared_mem = compute_properties.maxSharedLocalMemory;
  int max_group_size = compute_properties.maxTotalGroupSize;
  int num_subgroup_sizes = compute_properties.numSubGroupSizes;
  PyObject *subgroup_sizes = PyTuple_New(num_subgroup_sizes);
  for (int i = 0; i < num_subgroup_sizes; i++) {
    PyTuple_SetItem(subgroup_sizes, i,
                    PyLong_FromLong(compute_properties.subGroupSizes[i]));
  }

  uint32_t memoryCount = 0;
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, nullptr);
  auto pMemoryProperties = new ze_device_memory_properties_t[memoryCount];
  for (uint32_t mem = 0; mem < memoryCount; ++mem) {
    pMemoryProperties[mem].stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    pMemoryProperties[mem].pNext = nullptr;
  }
  zeDeviceGetMemoryProperties(phDevice, &memoryCount, pMemoryProperties);

  int mem_clock_rate = pMemoryProperties[0].maxClockRate;
  int mem_bus_width = pMemoryProperties[0].maxBusWidth;

  delete[] pMemoryProperties;

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:N}", "max_shared_mem",
                       max_shared_mem, "multiprocessor_count",
                       multiprocessor_count, "sm_clock_rate", sm_clock_rate,
                       "mem_clock_rate", mem_clock_rate, "mem_bus_width",
                       mem_bus_width, "max_work_group_size", max_group_size,
                       "sub_group_sizes", subgroup_sizes);
}

void freeKernel(PyObject *p) {
  delete reinterpret_cast<sycl::kernel *>(PyCapsule_GetPointer(p, "kernel"));
}

void freeKernelBundle(PyObject *p) {
  delete reinterpret_cast<
      sycl::kernel_bundle<sycl::bundle_state::executable> *>(
      PyCapsule_GetPointer(p, "kernel_bundle"));
}

using Spills = int32_t;

template <typename L0_DEVICE, typename L0_CONTEXT>
std::tuple<ze_module_handle_t, ze_kernel_handle_t, Spills>
compileLevelZeroObjects(uint8_t *binary_ptr, const size_t binary_size,
                        const std::string &kernel_name, L0_DEVICE l0_device,
                        L0_CONTEXT l0_context, const std::string &build_flags,
                        const bool is_spv) {
  std::cout << "# [mdziado][compileLevelZeroObjects][1] called before create_module" << std::endl;

  auto l0_module =
      checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                    binary_size, build_flags.data(), is_spv));
  std::cout << "# [mdziado][compileLevelZeroObjects][2] called before create_function" << std::endl;

  // Retrieve the kernel properties (e.g. register spills).
  auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));
  std::cout << "# [mdziado][compileLevelZeroObjects][3] called before zeKernelGetProperties" << std::endl;


  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  checkSyclErrors(
      std::make_tuple(NULL, zeKernelGetProperties(l0_kernel, &props)));

  const int32_t n_spills = props.spillMemSize;
  std::cout << "# [mdziado][compileLevelZeroObjects][4] called before return" << std::endl;
  return std::make_tuple(l0_module, l0_kernel, n_spills);
}

struct BuildFlags {
  std::string build_flags_str;

  const char *LARGE_GRF_FLAG{"-cl-intel-256-GRF-per-thread"};
  const char *SMALL_GRF_FLAG{"-cl-intel-128-GRF-per-thread"};
  const char *AUTO_GRF_FLAG{"-cl-intel-enable-auto-large-GRF-mode"};

  BuildFlags(const char *build_flags) : build_flags_str(build_flags) {}

  const std::string &operator()() const { return build_flags_str; }

  int32_t n_regs() const {
    if (build_flags_str.find(LARGE_GRF_FLAG) != std::string::npos) {
      return 256;
    }
    if (build_flags_str.find(SMALL_GRF_FLAG) != std::string::npos) {
      return 128;
    }
    // TODO: arguably we could return 128 if we find no flag instead of 0. For
    // now, stick with the conservative choice and alert the user only if a
    // specific GRF mode is specified.
    return 0;
  }

  const bool hasGRFSizeFlag() const {
    if (build_flags_str.find(LARGE_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(SMALL_GRF_FLAG) != std::string::npos ||
        build_flags_str.find(AUTO_GRF_FLAG) != std::string::npos) {
      return true;
    }

    return false;
  }

  void addLargeGRFSizeFlag() {
    build_flags_str = build_flags_str.append(" ").append(LARGE_GRF_FLAG);
  }
};

sycl::context get_default_context(const sycl::device &sycl_device) {
  const auto &platform = sycl_device.get_platform();
#ifdef WIN32
  sycl::context ctx;
  try {
#if __SYCL_COMPILER_VERSION >= 20250604
    ctx = platform.khr_get_default_context();
#else
    ctx = platform.ext_oneapi_get_default_context();
#endif
  } catch (const std::runtime_error &ex) {
    // This exception is thrown on Windows because
    // khr_get_default_context is not implemented. But it can be safely
    // ignored it seems.
#if _DEBUG
    std::cout << "ERROR: " << ex.what() << std::endl;
#endif
  }
  return ctx;
#else
#if __SYCL_COMPILER_VERSION >= 20250604
  return platform.khr_get_default_context();
#else
  return platform.ext_oneapi_get_default_context();
#endif
#endif
}

void segfault_handler(int sig) {
    void *array[32];
    size_t size = backtrace(array, 32);

    std::cout<< "\n# [mdziado][segfault_handler] Caught signal " << sig << " (SIGSEGV)" << std::endl;
    std::cout << "Stack trace (" << size << " frames):" << std::endl;

    // backtrace_symbols allocates memory and returns an array of strings
    char **symbols = backtrace_symbols(array, size);
    if (symbols) {
        for (size_t i = 0; i < size; ++i) {
            std::cout << "  [" << i << "] " << symbols[i] << std::endl;
        }
        free(symbols);
    } else {
        std::cout << "  <backtrace_symbols returned null>" << std::endl;
    }

    // It's unsafe to continue after SIGSEGV — exit immediately
    _exit(1);
}

extern "C" EXPORT_FUNC PyObject *load_binary(PyObject *args) {
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int is_spv;
  int devId;

  // Register signal handler for SIGSEGV
  struct sigaction sa{};
  sa.sa_handler = segfault_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;

  sigaction(SIGSEGV, &sa, nullptr);

  if (!PyArg_ParseTuple(args, "sSispi", &name, &py_bytes, &shared,
                        &build_flags_ptr, &is_spv, &devId)) {
    std::cout << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }
  std::cout << "# [mdziado][load_binary][1] called, name=" << name << ", is_spv=" << is_spv << ", devId=" << devId << std::endl;

  if (devId > g_sycl_l0_device_list.size()) {
    std::cout << "Device is not found " << std::endl;
    return NULL;
  }
  std::cout << "# [mdziado][load_binary][2] device found" << std::endl;

  BuildFlags build_flags(build_flags_ptr);

  try {

    const auto &sycl_l0_device_pair = g_sycl_l0_device_list[devId];
    const sycl::device sycl_device = sycl_l0_device_pair.first;
    const auto l0_device = sycl_l0_device_pair.second;

    const std::string kernel_name = name;
    const size_t binary_size = PyBytes_Size(py_bytes);

    uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
    const auto &ctx = get_default_context(sycl_device);
    const auto l0_context =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    ze_device_compute_properties_t compute_properties = {};
    compute_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
    zeDeviceGetComputeProperties(l0_device, &compute_properties);
    int32_t n_max_threads = compute_properties.maxTotalGroupSize;

    std::cout << "# [mdziado][load_binary][3-1] before compileLevelZeroObject" << std::endl;
    try {
    auto [l0_module, l0_kernel, n_spills] =
        compileLevelZeroObjects(binary_ptr, binary_size, kernel_name, l0_device,
                                l0_context, build_flags(), is_spv);
    } catch (std::exception& e) {
        std::cout << "# [mdziado][load_binary] exception catched, e=" << e.what() << std::endl;
        [[maybe_unused]] int i = 222;
        i += 3;
        i += 5;
        i += 7;
    }
    std::cout << "# [mdziado][load_binary][3-2] after compileLevelZeroObject" << std::endl;

    const bool debugEnabled = getBoolEnv("TRITON_DEBUG");



    if (is_spv) {
      constexpr int32_t max_reg_spill = 1000;
      const bool is_GRF_mode_specified = build_flags.hasGRFSizeFlag();

      // If the register mode isn't set, and the number of spills is greater
      // than the threshold, recompile the kernel using large GRF mode.
      if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
        if (debugEnabled)
          std::cout << "(I): Detected " << n_spills
                    << " spills, recompiling the kernel using large GRF mode"
                    << std::endl;

        build_flags.addLargeGRFSizeFlag();

        try {
          auto [l0_module_dgrf, l0_kernel_dgrf, n_spills_dgrf] =
              compileLevelZeroObjects(binary_ptr, binary_size, kernel_name,
                                      l0_device, l0_context, build_flags(),
                                      is_spv);

          if (debugEnabled)
            std::cout << "(I): Kernel has now " << n_spills_dgrf << " spills"
                      << std::endl;

          std::swap(l0_module, l0_module_dgrf);
          std::swap(l0_kernel, l0_kernel_dgrf);
          std::swap(n_spills, n_spills_dgrf);

          // clean up the unused module and kernel.
          auto error_no = zeKernelDestroy(l0_kernel_dgrf);
          if (error_no != ZE_RESULT_SUCCESS) {
            std::cout
                << "[Ignoring] Intel - Error during destroy unused L0 kernel"
                << std::endl;
          }
          error_no = zeModuleDestroy(l0_module_dgrf);
          if (error_no != ZE_RESULT_SUCCESS) {
            std::cout
                << "[Ignoring] Intel - Error during destroy unused L0 module"
                << std::endl;
          }
        } catch (const std::exception &e) {
          std::cout<< "[Ignoring] Error during Intel loadBinary with large "
                       "registers: "
                    << e.what() << std::endl;
          // construct previous working version
          build_flags = BuildFlags(build_flags_ptr);
        }
      }
    }

    if (debugEnabled && n_spills) {
      std::cout << "(I): Detected " << n_spills << " spills for  \""
                << kernel_name << "\"" << std::endl;
    }

    auto n_regs = build_flags.n_regs();

    auto mod = new sycl::kernel_bundle<sycl::bundle_state::executable>(
        sycl::make_kernel_bundle<sycl::backend::ext_oneapi_level_zero,
                                 sycl::bundle_state::executable>(
            {l0_module, sycl::ext::oneapi::level_zero::ownership::transfer},
            ctx));
    sycl::kernel *fun = new sycl::kernel(
        sycl::make_kernel<sycl::backend::ext_oneapi_level_zero>(
            {*mod, l0_kernel,
             sycl::ext::oneapi::level_zero::ownership::transfer},
            ctx));
    auto kernel_py =
        PyCapsule_New(reinterpret_cast<void *>(fun), "kernel", freeKernel);
    auto kernel_bundle_py = PyCapsule_New(reinterpret_cast<void *>(mod),
                                          "kernel_bundle", freeKernelBundle);

    std::cout << "# [mdziado][load_binary][4-1] returning PyBuildValue" << std::endl;
    return Py_BuildValue("(OOiii)", kernel_bundle_py, kernel_py, n_regs,
                         n_spills, n_max_threads);

  } catch (const std::exception &e) {
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, e.what());
    std::cout << "Error during Intel loadBinary: " << e.what() << std::endl;
    PyGILState_Release(gil_state);
    std::cout << "# [mdziado][load_binary][4-2] returning NULL" << std::endl;
    return NULL;
  }
}

extern "C" EXPORT_FUNC PyObject *init_devices(PyObject *cap) {
  void *queue = NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);

  auto sycl_context = sycl_queue->get_context();

  // Get sycl-device
  const std::vector<sycl::device> &sycl_devices = sycl_context.get_devices();

  // Retrieve l0 devices
  const uint32_t deviceCount = sycl_devices.size();
  for (uint32_t i = 0; i < deviceCount; ++i) {
    g_sycl_l0_device_list.push_back(std::make_pair(
        sycl_devices[i], sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                             sycl_devices[i])));
    // workaround to get opencl extensions
    const auto &name = sycl_devices[i].get_info<sycl::info::device::name>();
    sycl::device opencl_device([&](const sycl::device &dev) -> int {
      return (dev.get_backend() == sycl::backend::opencl &&
              dev.get_info<sycl::info::device::name>() == name)
                 ? 1
                 : -1;
    });
    sycl_opencl_device_list.push_back(opencl_device);
  }

  return Py_BuildValue("(i)", deviceCount);
}

extern "C" EXPORT_FUNC PyObject *wait_on_sycl_queue(PyObject *cap) {
  void *queue = NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  sycl_queue->wait();
  Py_RETURN_NONE;
}

extern "C" EXPORT_FUNC PyObject *has_opencl_extension(int device_id,
                                                      const char *extension) {
  if (device_id >= sycl_opencl_device_list.size()) {
    std::cerr << "Device is not found, extension " << extension << std::endl
              << std::flush;
    Py_RETURN_FALSE;
  }
  const sycl::device &device = sycl_opencl_device_list[device_id];

  if (sycl::opencl::has_extension(device, extension))
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}
