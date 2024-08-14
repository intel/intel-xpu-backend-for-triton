// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {passthrough = ["nounwind"]}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK:  [[EIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  [[DEST:%.*]] = llvm.alloca [[EIGHT]] x i16 : (i32) -> !llvm.ptr
  // CHECK-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT: [[COORD0:%.*]] = llvm.insertelement %arg4, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT: [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK-NEXT: llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, [[COORD1]], [[DEST]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_16r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_32r32x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<4xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_16r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_32b_16r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_32r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_16r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_8b_32r32x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_16r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_32r8x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transform_8b_32r16x4cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=4, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload_(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.call spir_funccc @_Z51intel_sub_group_2d_block_read_transpose_32b_16r8x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=true, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1UC_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<1, Cached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1UC_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1C_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Cached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1C_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Streaming, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1S_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Streaming, 0>, #triton_gen.load_cache_control<1, Cached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1S_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, InvalidateAfterRead, 0>, #triton_gen.load_cache_control<1, Cached, 0>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1IAR_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:        llvm.func @triton_gen.2Dblockload(
  // CHECK:          llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_8b_8r32x2cPU3AS1viiiDv2_iPt(
  // CHECK-NOT:        triton_gen.DecorationCacheControlINTEL
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

// CHECK:   llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i8
// CHECK-LABEL:   llvm.func @matrix_2Dblockload
llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:           [[ELEM_SIZE_IN_BITS:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[TILE_WIDTH:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:           [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[VBLOCKS:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:           [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i8({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ELEM_SIZE_IN_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi8>
  llvm.return
}

// -----

// COM: This case come from the 06 tutorial of FP8 flash attention.
// CHECK:    llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i8
// CHECK-LABEL:   llvm.func @matrix_2Dblockload
llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:           [[ELEM_SIZE_IN_BITS:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[TILE_WIDTH:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:           [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[VBLOCKS:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:           [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:           [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i8({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ELEM_SIZE_IN_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<32xi8>
  llvm.return
}

// -----

// CHECK:    llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i8
// CHECK-LABEL:   llvm.func @matrix_2Dblockload
llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:           [[ELEM_SIZE_IN_BITS:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[TILE_WIDTH:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK:           [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[VBLOCKS:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:           [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i8({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ELEM_SIZE_IN_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi8>
  llvm.return
}

// -----

// CHECK:   llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i16
// CHECK-LABEL:   llvm.func @matrix_2Dblockload
llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:           [[ELEM_SIZE_IN_BITS:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:           [[TILE_WIDTH:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK:           [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[VBLOCKS:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:           [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i16({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ELEM_SIZE_IN_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

// CHECK:    llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v1i32
// CHECK-LABEL:   llvm.func @matrix_2Dblockload
llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:           [[ELEM_SIZE_IN_BITS:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK:           [[TILE_WIDTH:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:           [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:           [[VBLOCKS:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:           [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:           [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v1i32({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[ELEM_SIZE_IN_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=2, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<1xi32>
  llvm.return
}
