//===-- TRITONGENAttrDefs.td - dialect attributes def.  ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_ATTRDEFS
#define TRITONGEN_ATTRDEFS

include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TritonGEN_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TritonGEN_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different subgroup reduce kinds.
def TritonGEN_ReduceKindAttr : I32EnumAttr<"ReduceKind", "TritonGEN subgroup reduce kind",
  [
    I32EnumAttrCase<"ADD",  0, "add">,
    I32EnumAttrCase<"MUL",  1, "mul">,
    I32EnumAttrCase<"MIN",  2, "min">,
    I32EnumAttrCase<"MAX",  3, "max">,
    I32EnumAttrCase<"AND",  4, "and">,
    I32EnumAttrCase<"OR",   5, "or">,
    I32EnumAttrCase<"XOR",  6, "xor">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different subgroup scan kinds.
def TritonGEN_ScanKindAttr : I32EnumAttr<"ScanKind", "TritonGEN subgroup scan kind",
  [
    I32EnumAttrCase<"INCLUSIVE",  0, "inclusive">,
    I32EnumAttrCase<"EXCLUSIVE",  1, "exclusive">,
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
    I32EnumAttrCase<"TF32",   10,  "tf32">,
    I32EnumAttrCase<"BF16",   11,  "bf16">,
    I32EnumAttrCase<"FP16",   12, "f16">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different cache control settings for a load operation.
def TritonGEN_LoadCacheControl : I32EnumAttr<"LoadCacheControl",
  "TritonGEN load cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"L1UC_L3UC",   1,  "L1UC_L3UC">, // Override to L1 uncached and L3 uncached
    I32EnumAttrCase<"L1UC_L3C",    2,  "L1UC_L3C">,  // Override to L1 uncached and L3 cached
    I32EnumAttrCase<"L1C_L3UC",    3,  "L1C_L3UC">,  // Override to L1 cached and L3 uncached
    I32EnumAttrCase<"L1C_L3C",     4,  "L1C_L3C">,   // Override to L1 cached and L3 cached
    I32EnumAttrCase<"L1S_L3UC",    5,  "L1S_L3UC">,  // Override to L1 streaming load and L3 uncached
    I32EnumAttrCase<"L1S_L3C",     6,  "L1S_L3C">,   // Override to L1 streaming load and L3 cached
    I32EnumAttrCase<"L1IAR_L3C",   7,  "L1IAR_L3C">  // Override to L1 invalidate-after-read, and L3 cached
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

/// Enum attribute of the different cache control settings for a store operation.
def TritonGEN_StoreCacheControl : I32EnumAttr<"StoreCacheControl",
  "TritonGEN store cache control",
  [
    I32EnumAttrCase<"DEFAULT",     0,  "Default">,
    I32EnumAttrCase<"L1UC_L3UC",   1,  "L1UC_L3UC">, // Override to L1 uncached and L3 uncached
    I32EnumAttrCase<"L1UC_L3WB",   2,  "L1UC_L3WB">, // Override to L1 uncached and L3 written back
    I32EnumAttrCase<"L1WT_L3UC",   3,  "L1WT_L3UC">, // Override to L1 written through and L3 uncached
    I32EnumAttrCase<"L1WT_L3WB",   4,  "L1WT_L3WB">, // Override to L1 written through and L3 written back
    I32EnumAttrCase<"L1S_L3UC",    5,  "L1S_L3UC">,  // Override to L1 streaming and L3 uncached
    I32EnumAttrCase<"L1S_L3WB",    6,  "L1S_L3WB">,  // Override to L1 streaming and L3 written back
    I32EnumAttrCase<"L1WB_L3WB",   7,  "L1WB_L3WB">  // Override to L1 written through and L3 written back
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

class TritonGEN_LoadStoreCacheControlDecoration<string loadOrStore>
    : TritonGEN_Attr<loadOrStore # "CacheControlDecoration", !tolower(loadOrStore) # "_cache_control"> {
  let summary = "An attribute specifying " # !tolower(loadOrStore) # " cache control";
  let description = [{
    A }] # !tolower(loadOrStore) # [{ cache control attribute has a one-to-one
    correspondance with the SPIR-V decoration shown in
    https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_cache_controls.asciidoc#decorations. The
    only differences are there is no need to add the SPIR-V decoration key, as
    that will be inferred from the attribute type, and an additional
    `operand_number` parameter is needed, as this is applied to pointer user
    operations, following
    https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/2587 design.
  }];
  let parameters = (ins "uint32_t":$cache_level,
                        loadOrStore # "CacheControlDecorationEnum":$cache_control,
                        "uint32_t":$operand_number);
  let assemblyFormat = "`<` $cache_level `,` $cache_control `,` $operand_number `>`";
}

/// Enum attribute for load cache controls.
///
/// See
/// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_cache_controls.asciidoc#Load_Cache_Control.
def TritonGEN_LoadCacheControlDecorationEnum : I32EnumAttr<"LoadCacheControlDecorationEnum",
  "TritonGEN load cache controls",
  [ I32EnumAttrCase<"Uncached",            0, "Uncached">,
    I32EnumAttrCase<"Cached",              1, "Cached">,
    I32EnumAttrCase<"Streaming",           2, "Streaming">,
    I32EnumAttrCase<"InvalidateAfterRead", 3, "InvalidateAfterRead">,
    I32EnumAttrCase<"ConstCached",         4, "ConstCached">
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

def TritonGEN_LoadCacheControlDecoration
    : TritonGEN_LoadStoreCacheControlDecoration<"Load">;

/// Enum attribute for store cache controls.
///
/// See
/// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/INTEL/SPV_INTEL_cache_controls.asciidoc#Store_Cache_Control.
def TritonGEN_StoreCacheControlDecorationEnum : I32EnumAttr<"StoreCacheControlDecorationEnum",
  "TritonGEN store cache controls",
  [ I32EnumAttrCase<"Uncached",     0, "Uncached">,
    I32EnumAttrCase<"WriteThrough", 1, "WriteThrough">,
    I32EnumAttrCase<"WriteBack",    2, "WriteBack">,
    I32EnumAttrCase<"Streaming",    3, "Streaming">,
  ]> {
  let cppNamespace = "::mlir::triton::TritonGEN";
}

def TritonGEN_StoreCacheControlDecoration
    : TritonGEN_LoadStoreCacheControlDecoration<"Store">;

def TritonGEN_CacheControlsDecoration
    : TritonGEN_Attr<"DecorationCacheControl", "decoration_cache_control"> {
  let summary = "An attribute specifying an operation cache controls";
  let description = [{
    Attribute corresponding to `!spirv.DecorationCacheControlINTEL` metadata
    described in
    https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/2587. This
    metadata is a list of `LoadCacheControlDecoration` and
    `StoreCacheControlDecoration` attributes specifying cache control
    information.

    The following MLIR code using this attribute:

    ```mlir
    llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr) {
      %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.store_cache_control<1, WriteThrough, 0>, #triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
      llvm.return
    }
    ```

    Will be translated to the following LLVM IR:

    ```llvm
    define void @triton_gen.cache_controls(ptr %0) {
      %1 = load i32, ptr %0, align 4, !spirv.DecorationCacheControlINTEL !1
      ret void
    }

    !1 = !{!2, !3, !4, !5}
    !2 = !{i32 6443, i32 0, i32 0, i32 0}
    !3 = !{i32 6443, i32 1, i32 1, i32 0}
    !4 = !{i32 6442, i32 0, i32 1, i32 0}
    !5 = !{i32 6442, i32 1, i32 0, i32 0}
    ```

    $decorations must be a non-empty list of cache controls attributes and two
    attributes of the same nature (load or store) in the list cannot be applied
    to the same cache level, as per the SPIR-V validation rules.
  }];
  let parameters = (ins ArrayRefParameter<"::mlir::Attribute">:$decorations);
  let assemblyFormat = "`<` $decorations `>`";
  let genVerifyDecl = 1;
}

#endif // TRITONGEN_ATTRDEFS
