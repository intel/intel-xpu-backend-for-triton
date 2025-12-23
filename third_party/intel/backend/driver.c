//===- driver.c -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
static std::vector<int> g_opencl_enabled;

template <typename T>
static inline T checkSyclErrors(const std::tuple<T, ze_result_t> tuple) {
  const auto code = std::get<1>(tuple);
  if (code != ZE_RESULT_SUCCESS) {
    throw std::runtime_error(parseZeResultCode(code));
  }
  return std::get<0>(tuple);
}


static void append_packed_string(std::vector<uint32_t> &words,
                                 const std::string &s) {
  // SPIR-V strings are null-terminated and packed little-endian into 32-bit
  // words.
  const size_t nbytes = s.size() + 1; // + '\0'
  const size_t nwords = (nbytes + 3) / 4;

  for (size_t w = 0; w < nwords; ++w) {
    uint32_t word = 0;
    for (size_t b = 0; b < 4; ++b) {
      const size_t idx = w * 4 + b;
      const uint8_t ch = (idx < s.size()) ? static_cast<uint8_t>(s[idx]) : 0;
      word |= (static_cast<uint32_t>(ch) << (8 * b));
    }
    words.push_back(word);
  }
}

static void emit_inst(std::vector<uint32_t> &out, uint16_t opcode,
                      std::initializer_list<uint32_t> ops) {
  const uint16_t wc = static_cast<uint16_t>(1 + ops.size());
  const uint32_t first = (static_cast<uint32_t>(wc) << 16) | opcode;
  out.push_back(first);
  out.insert(out.end(), ops.begin(), ops.end());
}

// Build a minimal SPIR-V "kernel" module that declares one OpExtension.
// We probe support by seeing whether zeModuleCreate accepts it.
static std::vector<uint32_t>
make_spirv_probe_module(const std::string &spirv_extension_name) {
  const uint32_t id_void_t = 1;
  const uint32_t id_fnty_void = 2;
  const uint32_t id_fn_main = 3;
  const uint32_t id_label = 4;
  const uint32_t bound = 5;

  std::vector<uint32_t> m;

  // ---- SPIR-V header ----
  m.push_back(0x07230203); // Magic
  m.push_back(0x00010000); // Version 1.0
  m.push_back(0);          // Generator (unknown)
  m.push_back(bound);      // Bound
  m.push_back(0);          // Schema

  // ---- Capabilities ----
  emit_inst(m, /*OpCapability*/ 17, {4}); // Addresses
  emit_inst(m, /*OpCapability*/ 17, {6}); // Kernel

  // ---- OpExtension "..." ----
  {
    std::vector<uint32_t> tmp;
    append_packed_string(tmp, spirv_extension_name);
    const uint16_t wc = static_cast<uint16_t>(1 + tmp.size());
    m.push_back((static_cast<uint32_t>(wc) << 16) | /*OpExtension*/ 10);
    m.insert(m.end(), tmp.begin(), tmp.end());
  }

  // ---- OpMemoryModel Physical64 OpenCL ----
  emit_inst(m, /*OpMemoryModel*/ 14, {2, 2});

  // ---- OpEntryPoint Kernel %3 "main" ----
  {
    std::vector<uint32_t> name_words;
    append_packed_string(name_words, "main");
    const uint16_t wc = static_cast<uint16_t>(1 + 2 + name_words.size());
    m.push_back((static_cast<uint32_t>(wc) << 16) | /*OpEntryPoint*/ 15);
    m.push_back(/*Kernel*/ 6);
    m.push_back(id_fn_main);
    m.insert(m.end(), name_words.begin(), name_words.end());
  }

  // ---- Types ----
  emit_inst(m, /*OpTypeVoid*/ 19, {id_void_t});
  emit_inst(m, /*OpTypeFunction*/ 33, {id_fnty_void, id_void_t});

  // ---- Function ----
  emit_inst(m, /*OpFunction*/ 54, {id_void_t, id_fn_main, 0, id_fnty_void});
  emit_inst(m, /*OpLabel*/ 248, {id_label});
  emit_inst(m, /*OpReturn*/ 253, {});
  emit_inst(m, /*OpFunctionEnd*/ 56, {});

  return m;
}

static bool probe_spirv_extension_by_module_create(ze_context_handle_t ctx,
                                                   ze_device_handle_t dev,
                                                   const char *ext_name) {
  std::vector<uint32_t> spirv = make_spirv_probe_module(ext_name);

  ze_module_desc_t desc = {};
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  desc.inputSize = spirv.size() * sizeof(uint32_t);
  desc.pInputModule = reinterpret_cast<const uint8_t *>(spirv.data());
  desc.pBuildFlags = "";

  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t log = nullptr;

  const ze_result_t r = zeModuleCreate(ctx, dev, &desc, &module, &log);

  if (log) {
    if (getBoolEnv("TRITON_DEBUG")) {
      size_t sz = 0;
      zeModuleBuildLogGetString(log, &sz, nullptr);
      std::string s(sz ? sz : 1, '\0');
      zeModuleBuildLogGetString(log, &sz, s.data());
      std::cerr << "(I): L0 SPIR-V probe log for \"" << ext_name
                << "\": " << s << std::endl;
    }
    zeModuleBuildLogDestroy(log);
  }

  if (module)
    zeModuleDestroy(module);

  return r == ZE_RESULT_SUCCESS;
}

static bool ze_driver_has_extension(ze_driver_handle_t driver,
                                    const char *ext_name) {
  uint32_t count = 0;
  ze_result_t r = zeDriverGetExtensionProperties(driver, &count, nullptr);
  if (r != ZE_RESULT_SUCCESS || count == 0)
    return false;

  std::vector<ze_driver_extension_properties_t> props(count);
  r = zeDriverGetExtensionProperties(driver, &count, props.data());
  if (r != ZE_RESULT_SUCCESS)
    return false;

  for (const auto &p : props) {
    if (std::string(p.name) == ext_name)
      return true;
  }

  return false;
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
  // To align with other backends - convert MHz to KHz
  int sm_clock_rate = device_properties.coreClockRate * 1000;

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

  // To align with other backends - convert MHz to KHz
  // https://github.com/intel/compute-runtime/blob/cfa007e5519d3a038d726b62237b86fca9a49e2c/shared/source/xe_hpc_core/linux/product_helper_pvc.cpp#L51
  int mem_clock_rate = pMemoryProperties[0].maxClockRate * 1000;
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
  auto l0_module =
      checkSyclErrors(create_module(l0_context, l0_device, binary_ptr,
                                    binary_size, build_flags.data(), is_spv));

  // Retrieve the kernel properties (e.g. register spills).
  auto l0_kernel = checkSyclErrors(create_function(l0_module, kernel_name));

  ze_kernel_properties_t props;
  props.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
  props.pNext = nullptr;
  checkSyclErrors(
      std::make_tuple(NULL, zeKernelGetProperties(l0_kernel, &props)));

  const int32_t n_spills = props.spillMemSize;

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
#if defined(_WIN32)
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

static BuildFlags last_build_flag("");

extern "C" EXPORT_FUNC PyObject *get_last_selected_build_flags() {
  return Py_BuildValue("s", last_build_flag().data());
}

extern "C" EXPORT_FUNC PyObject *load_binary(PyObject *args) {
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int is_spv;
  int devId;

  if (!PyArg_ParseTuple(args, "sSispi", &name, &py_bytes, &shared,
                        &build_flags_ptr, &is_spv, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  if (devId > g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

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

    auto [l0_module, l0_kernel, n_spills] =
        compileLevelZeroObjects(binary_ptr, binary_size, kernel_name, l0_device,
                                l0_context, build_flags(), is_spv);

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
            std::cerr
                << "[Ignoring] Intel - Error during destroy unused L0 kernel"
                << std::endl;
          }
          error_no = zeModuleDestroy(l0_module_dgrf);
          if (error_no != ZE_RESULT_SUCCESS) {
            std::cerr
                << "[Ignoring] Intel - Error during destroy unused L0 module"
                << std::endl;
          }
        } catch (const std::exception &e) {
          std::cerr << "[Ignoring] Error during Intel loadBinary with large "
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
    last_build_flag = build_flags;
    return Py_BuildValue("(OOiii)", kernel_bundle_py, kernel_py, n_regs,
                         n_spills, n_max_threads);

  } catch (const std::exception &e) {
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, e.what());
    std::cerr << "Error during Intel loadBinary: " << e.what() << std::endl;
    PyGILState_Release(gil_state);
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

  g_sycl_l0_device_list.clear();
  sycl_opencl_device_list.clear();
  g_opencl_enabled.clear();

  // Get all OpenCL devices (if any) once, then match by name.
  std::vector<sycl::device> opencl_devices;
  for (const auto &platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::opencl)
      continue;
    try {
      const auto &devices = platform.get_devices();
      opencl_devices.insert(opencl_devices.end(), devices.begin(),
                            devices.end());
    } catch (...) {
    }
  }

  // Retrieve L0 devices (from the PyTorch-provided SYCL context).
  const uint32_t deviceCount = sycl_devices.size();
  for (uint32_t i = 0; i < deviceCount; ++i) {
    g_sycl_l0_device_list.push_back(std::make_pair(
        sycl_devices[i], sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                             sycl_devices[i])));

    // Try to find a matching OpenCL SYCL device (same name). If not found,
    // OpenCL extension checks will be disabled for this device.
    const auto &name = sycl_devices[i].get_info<sycl::info::device::name>();
    auto it = std::find_if(
        opencl_devices.begin(), opencl_devices.end(),
        [&](const sycl::device &dev) {
          return dev.get_info<sycl::info::device::name>() == name;
        });

    if (it != opencl_devices.end()) {
      sycl_opencl_device_list.push_back(*it);
      g_opencl_enabled.push_back(1);
    } else {
      // Keep the indexing stable.
      sycl_opencl_device_list.push_back(sycl_devices[i]);
      g_opencl_enabled.push_back(0);
    }
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

extern "C" EXPORT_FUNC PyObject *is_opencl_enabled(int device_id) {
  if (device_id >= g_opencl_enabled.size())
    Py_RETURN_FALSE;
  if (g_opencl_enabled[device_id])
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

extern "C" EXPORT_FUNC PyObject *has_opencl_extension(int device_id,
                                                      const char *extension) {
  if (device_id >= sycl_opencl_device_list.size()) {
    std::cerr << "Device is not found, extension " << extension << std::endl
              << std::flush;
    Py_RETURN_FALSE;
  }

  if (device_id >= g_opencl_enabled.size() || !g_opencl_enabled[device_id]) {
    Py_RETURN_FALSE;
  }

  const sycl::device &device = sycl_opencl_device_list[device_id];

  if (device.get_backend() != sycl::backend::opencl)
    Py_RETURN_FALSE;

  if (sycl::opencl::has_extension(device, extension))
    Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

extern "C" EXPORT_FUNC PyObject *has_spirv_extension(int device_id,
                                                     const char *extension) {
  if (device_id >= g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found, extension " << extension << std::endl
              << std::flush;
    Py_RETURN_FALSE;
  }

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[device_id];
  const sycl::device sycl_device = sycl_l0_device_pair.first;
  const auto l0_device = sycl_l0_device_pair.second;

  // zeInit is typically performed under SYCL/PyTorch, but it's safe to call.
  zeInit(0);

  const auto &ctx = get_default_context(sycl_device);
  const auto l0_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  if (probe_spirv_extension_by_module_create(l0_context, l0_device, extension))
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

extern "C" EXPORT_FUNC PyObject *has_ze_extension(int device_id,
                                                  const char *extension) {
  if (device_id >= g_sycl_l0_device_list.size()) {
    std::cerr << "Device is not found, extension " << extension << std::endl
              << std::flush;
    Py_RETURN_FALSE;
  }

  const auto &sycl_l0_device_pair = g_sycl_l0_device_list[device_id];
  const sycl::device sycl_device = sycl_l0_device_pair.first;

  zeInit(0);

  const auto l0_driver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      sycl_device.get_platform());

  if (ze_driver_has_extension(l0_driver, extension))
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}
