#ifndef TT_LINK_INCLUDES
#define TT_LINK_INCLUDES

#include <stdint.h>

#include <assert.h>
#include <cstdint>
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <sycl/sycl.hpp>
#if defined(_WIN32)
#define EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC
#endif

typedef sycl::queue &TT_StreamTy;
typedef int32_t TT_ResultTy;

#define TT_ERROR_INVALID_VALUE ZE_RESULT_ERROR_INVALID_ARGUMENT

#endif
