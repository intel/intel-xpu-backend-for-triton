//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef INCLUDE_PTI_H_
#define INCLUDE_PTI_H_

#include "pti/pti_export.h"
#include "pti/pti_version.h"

/* clang-format off */
#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @brief Return/Error codes
 */
typedef enum {
  PTI_SUCCESS = 0,                        //!< success
  PTI_STATUS_END_OF_BUFFER = 1,           //!< end of buffer reached, e.g., in ptiViewGetNextRecord
  PTI_ERROR_NOT_IMPLEMENTED = 2,          //!< functionality not implemented
  PTI_ERROR_BAD_ARGUMENT = 3,             //!< error code for invalid arguments
  PTI_ERROR_NO_CALLBACKS_SET = 4,         //!< error due to no callbacks set via ptiViewSetCallbacks
  PTI_ERROR_EXTERNAL_ID_QUEUE_EMPTY = 5,  //!< empty external ID-queue while working with
                                          //!< PTI_VIEW_EXTERNAL_CORRELATION
  PTI_ERROR_BAD_TIMESTAMP = 6,  //!< error in timestamp conversion, might be related with the user
                                //!< provided TimestampCallback
  PTI_ERROR_DRIVER = 50,        //!< unknown driver error
  PTI_ERROR_TRACING_NOT_INITIALIZED = 51,  //!< installed driver requires tracing enabling with
                                           //!< setting environment variable ZE_ENABLE_TRACING_LAYER
                                           //!< to 1
  PTI_ERROR_L0_LOCAL_PROFILING_NOT_SUPPORTED = 52,  //!< no Local profiling support in the installed
                                                    //!< driver

  PTI_ERROR_INTERNAL = 200  //!< internal error
} pti_result;

/**
 * @brief Helper function to return stringified enum members for pti_result
 *
 * @return const char*
 */
PTI_EXPORT const char* ptiResultTypeToString(pti_result result_value);

#if defined(__cplusplus)
}
#endif

#endif  // INCLUDE_PTI_H_
