// RUN: TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @__builtin_IB_subgroup_block_read_ap_u8_m8k32v1(!llvm.ptr {llvm.nonnull}, i32, i32, i32) -> vector<8xi16> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, will_return}
// CHECK: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(!llvm.ptr {llvm.nonnull}, i32) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, will_return}
// CHECK: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(!llvm.ptr {llvm.nonnull}, i32) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, no_unwind, will_return}
// CHECK: llvm.func spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload(i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  [[PTR:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK-DAG:  [[WIDTH:%.*]] = llvm.sub %arg1, [[ONE]] : i32
  // CHECK-DAG:  [[HEIGHT:%.*]] = llvm.sub %arg2, [[ONE]] : i32
  // CHECK-DAG:  [[PITCH:%.*]] = llvm.sub %arg3, [[ONE]] : i32
  // CHECK-DAG:  [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG:  [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[AP:%.*]] = llvm.call spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload([[PTR]], [[WIDTH]], [[HEIGHT]], [[PITCH]], [[ZERO]], [[ZERO]], [[C32]], [[C8]], [[C1]]) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr
  // CHECK:      llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX([[AP]], %arg4) {{.*}} : (!llvm.ptr, i32) -> ()
  // CHECK:      llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY([[AP]], %arg5) {{.*}} : (!llvm.ptr, i32) -> ()
  // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:             llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_u8_m8k32v1([[AP]], [[ZERO]], [[ZERO]], [[ZERO]]) {{.*}} : (!llvm.ptr, i32, i32, i32) -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_transpose_u32_m16k8v1
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=true, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_transform_u8_m32k16v1
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}
