//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef INCLUDE_PTI_VERSION_H_
#define INCLUDE_PTI_VERSION_H_

#include <stdint.h>

#include "pti/pti_export.h"

/* clang-format off */
#if defined(__cplusplus)
extern "C" {
#endif

#if !defined(PTI_VERSION)
#define PTI_VERSION 0.14.0
#endif

#define PTI_VERSION_STRING "0.14.0"
#define PTI_VERSION_MAJOR 0
#define PTI_VERSION_MINOR 14
#define PTI_VERSION_PATCH 0

typedef struct pti_version {
  uint32_t _major;
  uint32_t _minor;
  uint32_t _patch;
} pti_version;

/**
 * @brief Returns the compiled version of Intel(R) PTI
 *
 * @return c-string with compiled version of Intel(R) PTI
 */
PTI_EXPORT const char* ptiVersionString();

/**
 * @brief Returns the compiled version of Intel(R) PTI
 *
 * @return pti_version struct with compiled version of Intel(R) PTI
 */
pti_version PTI_EXPORT ptiVersion();

#if defined(__cplusplus)
}
#endif

#endif  // INCLUDE_PTI_VERSION_H_
