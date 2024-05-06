//===-- TRITONGENAttrDefs.td - dialect attributes def.  ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_ATTRDEFS
#define TRITONGEN_ATTRDEFS

include "mlir/IR/EnumAttr.td"

/// Enum attribute of the different shuffle kinds.
def TritonGEN_ShflKindAttr : I32EnumAttr<"ShflKind", "TritonGEN shuffle kind",
  [
    I32EnumAttrCase<"XOR",  0, "xor">,
    I32EnumAttrCase<"UP",   1, "up">,
    I32EnumAttrCase<"DOWN", 2, "down">,
    I32EnumAttrCase<"IDX",  3, "idx">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different floating-point rounding modes.
def TritonGEN_RoundingModeAttr : I32EnumAttr<"RoundingMode",
  "TritonGEN floating-point rounding mode",
  [
    I32EnumAttrCase<"UNUSED", 0, "unused">,
    I32EnumAttrCase<"RTE",    1, "rte">, ///< Round to nearest, ties to even
    I32EnumAttrCase<"RTN",    2, "rtn">, ///< Round toward negative
    I32EnumAttrCase<"RTP",    3, "rtp">, ///< Round toward positive
    I32EnumAttrCase<"RTZ",    4, "rtz">, ///< Round toward zero
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different precision types.
def TritonGEN_PrecisionTypeAttr : I32EnumAttr<"PrecisionType",
  "TritonGEN precision type",
  [
    I32EnumAttrCase<"UNUSED", 0,  "unused">,
    I32EnumAttrCase<"U8",     1,  "u8">,
    I32EnumAttrCase<"U4",     2,  "u4">,
    I32EnumAttrCase<"U2",     3,  "u2">,
    I32EnumAttrCase<"S8",     4,  "i8">,
    I32EnumAttrCase<"S4",     5,  "i4">,
    I32EnumAttrCase<"S2",     6,  "i2">,
    I32EnumAttrCase<"BF8",    7,  "bf8">,
    I32EnumAttrCase<"TF32",   8,  "tf32">,
    I32EnumAttrCase<"BF16",   9,  "bf16">,
    I32EnumAttrCase<"FP16",   10, "f16">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different cache control settings for a prefetch operation.
def TritonGEN_PrefetchCacheControl : I32EnumAttr<"PrefetchCacheControl",
  "TritonGEN prefetch cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"L1UC_L3UC",   1,  "L1UC_L3UC">,
    I32EnumAttrCase<"L1UC_L3C",    2,  "L1UC_L3C">,
    I32EnumAttrCase<"L1C_L3UC",    3,  "L1C_L3UC">,
    I32EnumAttrCase<"L1C_L3C",     4,  "L1C_L3C">,
    I32EnumAttrCase<"L1S_L3UC",    5,  "L1S_L3UC">,
    I32EnumAttrCase<"L1S_L3C",     6,  "L1S_L3C">,
    I32EnumAttrCase<"L1IAR_L3C",   7,  "L1IAR_L3C">,
    I32EnumAttrCase<"L1UC_L3CC",   8,  "L1UC_L3CC">,
    I32EnumAttrCase<"L1C_L3CC",    9,  "L1C_L3CC">,
    I32EnumAttrCase<"L1IAR_L3IAR", 10, "L1IAR_L3IAR">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different memory fence settings.
def TritonGEN_MemFence : I32EnumAttr<"MemFence",
  "TritonGEN memory fence",
  [ I32EnumAttrCase<"NONE",             0, "None">,
    I32EnumAttrCase<"LOCAL",            1, "Local">,
    I32EnumAttrCase<"GLOBAL",           2, "Global">,
    I32EnumAttrCase<"LOCAL_AND_GLOBAL", 3, "LocalAndGlobal">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different memory scope settings.
def TritonGEN_MemScope : I32EnumAttr<"MemScope",
  "TritonGEN memory scope",
  [ I32EnumAttrCase<"WORK_ITEM",       0, "WorkItem">,
    I32EnumAttrCase<"WORK_GROUP",      1, "WorkGroup">,
    I32EnumAttrCase<"DEVICE",          2, "Device">,
    I32EnumAttrCase<"ALL_SVM_DEVICES", 3, "AllSvmDevices">,
    I32EnumAttrCase<"SUB_GROUP",       4, "SubGroup">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

#endif // TRITONGEN_ATTRDEFS
