//===- arch_parser.c ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *parseDeviceArch(PyObject *self, PyObject *args) {
  uint64_t dev_arch;
  if (!PyArg_ParseTuple(args, "K", &dev_arch))
    return NULL;

  sycl::ext::oneapi::experimental::architecture sycl_arch =
      static_cast<sycl::ext::oneapi::experimental::architecture>(dev_arch);
  PyObject *tritonIntelGPUPassesModule =
      PyImport_ImportModule("triton._C.libtriton.intel.passes.ttgpuir");
  if (!tritonIntelGPUPassesModule) {
    PyErr_Print();
    printf("Error importing triton._C.libtriton.intel.passes.ttgpuir\n");
    return NULL;
  }
  PyObject *device_archs =
      PyObject_GetAttrString(tritonIntelGPUPassesModule, (char *)"DEVICE_ARCH");
  if (!device_archs) {
    Py_DECREF(tritonIntelGPUPassesModule);
    PyErr_Print();
    printf("Error unknown 'DEVICE_ARCH' attribute\n");
    return NULL;
  }
  PyObject *device_arch =
      PyObject_GetAttrString(device_archs, (char *)"UNKNOWN");

  switch (sycl_arch) {
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc:
    Py_DECREF(device_arch);
    device_arch = PyObject_GetAttrString(device_archs, (char *)"PVC");
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_dg2_g10:
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_dg2_g11:
    Py_DECREF(device_arch);
    device_arch = PyObject_GetAttrString(device_archs, (char *)"ATS");
    break;
  default:; // fall through
  }
  Py_DECREF(tritonIntelGPUPassesModule);
  Py_DECREF(device_archs);

  return Py_BuildValue("N", device_arch);
}

static PyMethodDef ModuleMethods[] = {
    {"parse_device_arch", parseDeviceArch, METH_VARARGS,
     "parse device architecture"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "arch_utils",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_arch_utils(void) {
  if (PyObject *m = PyModule_Create(&ModuleDef)) {
    PyModule_AddFunctions(m, ModuleMethods);
    return m;
  }
  return NULL;
}
