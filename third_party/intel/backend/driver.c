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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "sycl_functions.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static std::vector<sycl::device> g_sycl_device_list;

static PyObject *getDeviceProperties(PyObject *self, PyObject *args) {
  int device_id;
  if (!PyArg_ParseTuple(args, "i", &device_id))
    return NULL;

  if (device_id > g_sycl_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  const auto sycl_device = g_sycl_device_list[device_id];
  int sycl_max_shared_mem = sycl_device.get_info<sycl::info::device::local_mem_size>();
  int num_slices = sycl_device.get_info<sycl::ext::intel::info::device::gpu_slices>();
  int num_subslices_per_slice = sycl_device.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  int max_compute_units = num_slices * num_subslices_per_slice;
  int sycl_mem_clock_rate = sycl_device.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
  int sycl_mem_bus_width = sycl_device.get_info<sycl::ext::intel::info::device::memory_bus_width>();
  int sycl_max_wg_size = sycl_device.get_info<sycl::info::device::max_work_group_size>();
  auto sg_sizes = sycl_device.get_info<sycl::info::device::sub_group_sizes>();
  int sycl_max_clock_freq = sycl_device.get_info<sycl::info::device::max_clock_frequency>();

  PyObject *subgroup_sizes = PyTuple_New(sg_sizes.size());
  for (int i = 0; i < sg_sizes.size(); i++) {
    PyTuple_SetItem(subgroup_sizes, i,
                    PyLong_FromLong(sg_sizes[i]));
  }

  return Py_BuildValue("{s:i, s:i, s:i, s:i, s:i, s:i, s:N}", "max_shared_mem",
                       sycl_max_shared_mem, "multiprocessor_count",
                       max_compute_units, "sm_clock_rate", sycl_max_clock_freq,
                       "mem_clock_rate", sycl_mem_clock_rate, "mem_bus_width",
                       sycl_mem_bus_width, "max_work_group_size", sycl_max_wg_size,
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

// todo: add me
// struct BuildFlags2 {
//   std::vector<std::string> build_flags;

//   BuildFlags(const std::string& flags) {
//     auto split = [](const std::string& str, const string& delim) -> std::vector<std::string> {
//       std::vector<std::string> res;
//       size_t start{0}, end;
//       while((end = str.find(delim, start)) != std::string::npos) {
//         if (str.substr(start, end-start).size())
//             res.push_back(str.substr(start, end-start));
//         start = end+delim.size();
//       }
//       if (str.substr(start, end-start).size())
//         res.push_back(str.substr(start, end - start));
//       return res;
//     };
//     build_flags = split(flags, " ");
//   }

//   int32_t n_regs() const {
//     if (build_flags.count(LARGE_GRF_FLAG)) {
//       return 256;
//     }
//     if (build_flags.count(SMALL_GRF_FLAG)) {
//       return 128;
//     }
//     // TODO: arguably we could return 128 if we find no flag instead of 0. For
//     // now, stick with the conservative choice and alert the user only if a
//     // specific GRF mode is specified.
//     return 0;
//   }

//   const bool hasGRFSizeFlag() const {
//     return (build_flags.count(LARGE_GRF_FLAG) ||
//         build_flags.count(SMALL_GRF_FLAG) ||
//         build_flags.count(AUTO_GRF_FLAG)); 
  
//   void addLargeGRFSizeFlag() {
//     build_flags.push_back(LARGE_GRF_FLAG);
//   }
// };

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
    ctx = platform.ext_oneapi_get_default_context();
  } catch (const std::runtime_error &ex) {
    // This exception is thrown on Windows because
    // ext_oneapi_get_default_context is not implemented. But it can be safely
    // ignored it seems.
#if _DEBUG
    std::cout << "ERROR: " << ex.what() << std::endl;
#endif
  }
  return ctx;
#else
  return platform.ext_oneapi_get_default_context();
#endif
}

static std::vector<std::byte> toBytes(uint8_t* ptr, size_t size) {
    std::vector<std::byte> bytes(size);
    std::transform(ptr, ptr+size, bytes.begin(),
                   [](uint8_t c) { return std::byte(c); });
    return bytes;
}


static PyObject *loadBinary(PyObject *self, PyObject *args) {
  const char *name, *build_flags_ptr;
  int shared;
  PyObject *py_bytes;
  int devId;

  if (!PyArg_ParseTuple(args, "sSisi", &name, &py_bytes, &shared,
                        &build_flags_ptr, &devId)) {
    std::cerr << "loadBinary arg parse failed" << std::endl;
    return NULL;
  }

  if (devId > g_sycl_device_list.size()) {
    std::cerr << "Device is not found " << std::endl;
    return NULL;
  }

  BuildFlags build_flags(build_flags_ptr);

  try {

    const auto &sycl_l0_device_pair = g_sycl_device_list[devId];
    const sycl::device sycl_device = sycl_l0_device_pair;

    const std::string kernel_name = name;
    const size_t binary_size = PyBytes_Size(py_bytes);

    uint8_t *binary_ptr = (uint8_t *)PyBytes_AsString(py_bytes);
    const auto &ctx = get_default_context(sycl_device);

    const auto use_native_code =
        isEnvValueBool(getStrEnv("TRITON_XPU_GEN_NATIVE_CODE"));
    const bool is_spv = use_native_code ? !(*use_native_code) : true;
    assert(is_spv && "native codegen is disabled in this build");

    namespace syclex = sycl::ext::oneapi::experimental;
    std::vector<std::byte> spv = toBytes(binary_ptr, binary_size);
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
      syclex::create_kernel_bundle_from_source(
          ctx, syclex::source_language::spirv, spv); 

    // TODO: add build args
    // if (is_spv) {
    //   constexpr int32_t max_reg_spill = 1000;
    //   const bool is_GRF_mode_specified = build_flags.hasGRFSizeFlag();

    //   // If the register mode isn't set, and the number of spills is greater
    //   // than the threshold, recompile the kernel using large GRF mode.
    //   if (!is_GRF_mode_specified && n_spills > max_reg_spill) {
    //     const std::optional<bool> debugEnabled =
    //         isEnvValueBool(getStrEnv("TRITON_DEBUG"));
    //     if (debugEnabled)
    //       std::cout << "(I): Detected " << n_spills
    //                 << " spills, recompiling the kernel using large GRF mode"
    //                 << std::endl;

    //     build_flags.addLargeGRFSizeFlag();

    //     try {
    //       auto [l0_module, l0_kernel, n_spills] = compileLevelZeroObjects(
    //           binary_ptr, binary_size, kernel_name, l0_device, l0_context,
    //           build_flags(), is_spv);

    //       if (debugEnabled)
    //         std::cout << "(I): Kernel has now " << n_spills << " spills"
    //                   << std::endl;
    //     } catch (const std::exception &e) {
    //       std::cerr << "[Ignoring] Error during Intel loadBinary with large "
    //                    "registers: "
    //                 << e.what() << std::endl;
    //       // construct previous working version
    //       build_flags = BuildFlags(build_flags_ptr);
    //     }
    //   }
    // }

    auto n_regs = build_flags.n_regs();

    std::string log;
    std::vector<std::string> flags{build_flags()}; // todo: split the string
    sycl::kernel_bundle<sycl::bundle_state::executable> *mod =
        new sycl::kernel_bundle<sycl::bundle_state::executable>(syclex::build(
            kb_src, syclex::properties{syclex::build_options{flags},
                                       syclex::save_log{&log}}));

    auto fun = new sycl::kernel(mod->ext_oneapi_get_kernel(kernel_name));
    auto kernel_py =
        PyCapsule_New(reinterpret_cast<void *>(fun), "kernel", freeKernel);
    auto kernel_bundle_py = PyCapsule_New(reinterpret_cast<void *>(mod),
                                          "kernel_bundle", freeKernelBundle);
    Spills n_spills{}; // todo: needs a sycl equivalent

    sycl::kernel* kernel_ptr = reinterpret_cast<sycl::kernel*>(PyCapsule_GetPointer(kernel_py, "kernel"));
    assert(kernel_ptr && "kernel is null!");
    std::string kernel_name_2 = kernel_ptr->get_info<sycl::info::kernel::function_name>();

    return Py_BuildValue("(OOii)", kernel_bundle_py, kernel_py, n_regs,
                         n_spills);

  } catch (const std::exception &e) {
    char err[1024] = {0};
    std::string_view error_str(e.what());
    strncat(err, error_str.data(), std::min(error_str.size(), size_t(1024)));
    PyGILState_STATE gil_state;
    gil_state = PyGILState_Ensure();
    PyErr_SetString(PyExc_RuntimeError, err);
    std::cerr << "Error during Intel loadBinary: " << err << std::endl;
    PyGILState_Release(gil_state);
    return NULL;
  }
}

static PyObject *initContext(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  sycl::context context = sycl_queue->get_context();
  return Py_BuildValue("(K)", (uint64_t)(&context));
}

static PyObject *initDevices(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);

  auto sycl_context = sycl_queue->get_context();

  // Get sycl-device
  g_sycl_device_list = sycl_context.get_devices();
  const uint32_t deviceCount = g_sycl_device_list.size();

  return Py_BuildValue("(i)", deviceCount);
}

static PyObject *waitOnSYCLQueue(PyObject *self, PyObject *args) {
  PyObject *cap;
  void *queue = NULL;
  if (!PyArg_ParseTuple(args, "O", &cap))
    return NULL;
  if (!(queue = PyLong_AsVoidPtr(cap)))
    return NULL;
  sycl::queue *sycl_queue = static_cast<sycl::queue *>(queue);
  sycl_queue->wait();

  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
    {"load_binary", loadBinary, METH_VARARGS,
     "Load provided SPV into ZE driver"},
    {"get_device_properties", getDeviceProperties, METH_VARARGS,
     "Get the properties for a given device"},
    {"init_context", initContext, METH_VARARGS,
     "Initialize the ZE GPU context"},
    {"init_devices", initDevices, METH_VARARGS,
     "Initialize the ZE GPU devices and return device count"},
    {"wait_on_sycl_queue", waitOnSYCLQueue, METH_VARARGS,
     "call wait on a specific sycl::queue"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "spirv_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_spirv_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
