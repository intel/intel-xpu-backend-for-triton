// RUN: TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s --check-prefixes=CHECK,CHECK-COMMON
// RUN: TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 TRITONGEN_FORCE_GENISA=1 triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s --check-prefixes=CHECK-GENISA,CHECK-COMMON

// CHECK: llvm.func spir_funccc @__builtin_IB_subgroup_block_read_ap_u8_m8k32v1(!llvm.ptr {llvm.nonnull}, i32, i32, i32) -> vector<8xi16> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-GENISA:  llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v8i16.p0i8(!llvm.ptr {llvm.nonnull}, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
// CHECK-COMMON: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(!llvm.ptr {llvm.nonnull}, i32) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-COMMON: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(!llvm.ptr {llvm.nonnull}, i32) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, will_return}
// CHECK-COMMON: llvm.func spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload(i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COMMON:     llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-COMMON-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-COMMON-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-COMMON-DAG:  [[PTR:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK-COMMON-DAG:  [[WIDTH:%.*]] = llvm.sub %arg1, [[ONE]] : i32
  // CHECK-COMMON-DAG:  [[HEIGHT:%.*]] = llvm.sub %arg2, [[ONE]] : i32
  // CHECK-COMMON-DAG:  [[PITCH:%.*]] = llvm.sub %arg3, [[ONE]] : i32
  // CHECK-COMMON-DAG:  [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-COMMON-DAG:  [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-COMMON-DAG:  [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-COMMON:      [[AP:%.*]] = llvm.call spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload([[PTR]], [[WIDTH]], [[HEIGHT]], [[PITCH]], [[ZERO]], [[ZERO]], [[C32]], [[C8]], [[C1]]) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr
  // CHECK-COMMON:      llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX([[AP]], %arg4) {{.*}} : (!llvm.ptr, i32) -> ()
  // CHECK-COMMON:      llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY([[AP]], %arg5) {{.*}} : (!llvm.ptr, i32) -> ()
  // CHECK-COMMON:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:             llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_u8_m8k32v1([[AP]], [[ZERO]], [[ZERO]], [[ZERO]]) {{.*}} : (!llvm.ptr, i32, i32, i32) -> vector<8xi16>
  // CHECK-GENISA:      llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v8i16.p0i8([[AP]], [[ZERO]], [[ZERO]], {{.*}}) {{.*}} : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_transpose_u32_m16k8v1
  // CHECK-GENISA-DAG: [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
  // CHECK-GENISA-DAG: [[FALSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK-GENISA: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v8i32.p0i8({{.*}}, [[TRUE]], [[FALSE]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=true, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_transform_u8_m32k16v1
  // CHECK-GENISA-DAG: [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
  // CHECK-GENISA-DAG: [[FALSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK-GENISA: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v8i32.p0i8({{.*}}, [[FALSE]], [[TRUE]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
