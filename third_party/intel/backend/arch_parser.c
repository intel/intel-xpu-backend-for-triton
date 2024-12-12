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
  assert(PyArg_ParseTuple(args, "K", &dev_arch) && "Expected an integer");

  sycl::ext::oneapi::experimental::architecture sycl_arch =
      static_cast<sycl::ext::oneapi::experimental::architecture>(dev_arch);
  // FIXME: Add support for more architectures.
  std::string arch = "";
  switch (sycl_arch) {
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc:
    arch = "pvc";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21:
    arch = "bmg";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_lnl_m:
    arch = "lnl";
    break;
  default:
    printf("sycl_arch = %d", sycl_arch);
  }

  return Py_BuildValue("s", arch.c_str());
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
