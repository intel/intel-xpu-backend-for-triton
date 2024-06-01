//===- TritonGENAttrs.cpp - TritonGEN Attributes Definition --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::TritonGEN {
//===----------------------------------------------------------------------===//
// triton_gen.decoration_cache_control
//===----------------------------------------------------------------------===//

LogicalResult TritonGEN::DecorationCacheControlAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<Attribute> decorations) {
  llvm::SmallSet<uint32_t, 3> loadCacheLevels;
  llvm::SmallSet<uint32_t, 3> storeCacheLevels;
  for (Attribute attr : decorations) {
    LogicalResult res =
        TypeSwitch<Attribute, LogicalResult>(attr)
            .Case<LoadCacheControlDecorationAttr,
                  StoreCacheControlDecorationAttr>([emitError, &loadCacheLevels,
                                                    &storeCacheLevels](
                                                       auto attr)
                                                       -> LogicalResult {
              llvm::SmallSet<uint32_t, 3> &cacheLevels =
                  std::is_same_v<decltype(attr), LoadCacheControlDecorationAttr>
                      ? loadCacheLevels
                      : storeCacheLevels;
              if (!cacheLevels.insert(attr.getCacheLevel()).second)
                return emitError()
                       << "'triton_gen.decoration_cache_controls' cannot "
                          "specify more than one cache control decoration of "
                          "the same nature for the same cache level";
              return success();
            })
            .Default([emitError](Attribute attr) -> LogicalResult {
              return emitError()
                     << "'triton_gen.decoration_cache_controls' only accepts "
                        "LoadCacheControlDecorationAttr and "
                        "StoreCacheControlDecorationAttr attributes";
            });
    if (failed(res))
      return res;
  }
  return success();
}
} // namespace mlir::triton::TritonGEN
