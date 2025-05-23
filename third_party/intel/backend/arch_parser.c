//===- arch_parser.c ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <sycl/sycl.hpp>

#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __attribute__((visibility("default")))
#endif

extern "C" EXPORT_FUNC const char *parse_device_arch(uint64_t dev_arch) {
  sycl::ext::oneapi::experimental::architecture sycl_arch =
      static_cast<sycl::ext::oneapi::experimental::architecture>(dev_arch);
  // FIXME: Add support for more architectures.
  const char *arch = "";
  switch (sycl_arch) {
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_arl_h:
    arch = "arl";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21:
    arch = "bmg";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_dg2_g10:
    arch = "dg2";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_lnl_m:
    arch = "lnl";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_mtl_h:
    arch = "mtl";
    break;
  case sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc:
    arch = "pvc";
    break;
  default:
    std::cerr << "sycl_arch not recognized: " << (int)sycl_arch << std::endl;
  }

  return arch;
}
