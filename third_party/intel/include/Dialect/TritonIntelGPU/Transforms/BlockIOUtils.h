//===- BlockIOUtils.h - 2D Block IO tile utilities ---------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for 2D block I/O tile size computation and validation.
// Used by both the TTGIR LowerTo2DBlockLoad pass and the LLVM lowering.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "triton/Analysis/AxisInfo.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SetVector.h"
#include <optional>

namespace mlir::triton::gpu::intel {

/// Information about a 2D block I/O tile shape, computed from a LinearLayout.
struct BlockIOTileSizeInfo {
  BlockIOTileSizeInfo() = delete;
  BlockIOTileSizeInfo(int tileHeight, int tileWidth, int numElemPerPackedVal,
                      int vBlocks, int rowDim, int colDim, bool transpose,
                      std::optional<SetVector<unsigned>> regPackedBases)
      : tileHeight(tileHeight), tileWidth(tileWidth),
        numElemPerPackedVal(numElemPerPackedVal), vBlocks(vBlocks),
        rowDim(rowDim), colDim(colDim), transpose(transpose),
        regPackedBases(regPackedBases) {}
  static BlockIOTileSizeInfo unknown() {
    return {-1, -1, -1, -1, -1, -1, false, std::nullopt};
  }

  int tileHeight;
  int tileWidth;
  int numElemPerPackedVal;
  int vBlocks;
  int rowDim;
  int colDim;
  bool transpose;
  std::optional<SetVector<unsigned>> regPackedBases;

  bool isValid() const {
    return tileHeight >= 0 && tileWidth >= 0 && numElemPerPackedVal >= 0 &&
           vBlocks >= 0 && rowDim >= 0 && colDim >= 0;
  }
};

/// Compute the 2D block I/O tile shape from a LinearLayout.
/// Returns BlockIOTileSizeInfo::unknown() if the layout does not support
/// 2D block I/O.
template <bool IS_LOAD>
BlockIOTileSizeInfo
getBlockIOTileSize(const LinearLayout &ll, unsigned memContiguousDim,
                   unsigned elemSizeInBits, AxisInfo *maskAxisInfo = nullptr,
                   bool oneMatrixPerLoadForBT = false);

// Explicit instantiation declarations.
extern template BlockIOTileSizeInfo
getBlockIOTileSize<true>(const LinearLayout &, unsigned, unsigned, AxisInfo *,
                         bool);
extern template BlockIOTileSizeInfo
getBlockIOTileSize<false>(const LinearLayout &, unsigned, unsigned, AxisInfo *,
                          bool);

/// Check whether the packed element size and tile width satisfy the
/// 2D block address payload hardware restrictions.
bool check2DBlockAddressPayloadRestriction(unsigned packedElemSizeInBits,
                                           unsigned tileWidth);

} // namespace mlir::triton::gpu::intel
