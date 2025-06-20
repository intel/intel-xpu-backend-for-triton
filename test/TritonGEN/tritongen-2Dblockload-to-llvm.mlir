// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK:       [[EIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  [[DEST:%.*]] = llvm.alloca [[EIGHT]] x i16 : (i32) -> !llvm.ptr
  // CHECK-NEXT:  [[PTRTOINT:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK-NEXT:  [[VAL_63:%.*]] = llvm.mlir.constant(-64 : i64) : i64
  // CHECK-NEXT:  [[VAL_64:%.*]] = llvm.and [[PTRTOINT]], [[VAL_63]] : i64
  // CHECK-NEXT:  [[BASE_ALIGNED:%.*]] = llvm.inttoptr [[VAL_64]] : i64 to !llvm.ptr<1>
  // CHECK-NEXT:  [[CL:%.*]] = llvm.mlir.constant(63 : i64) : i64
  // CHECK-NEXT:  [[AND:%.*]] = llvm.and [[PTRTOINT]], [[CL]] : i64
  // CHECK-NEXT:  [[TRUNC:%.*]] = llvm.trunc [[AND]] : i64 to i32
  // CHECK-NEXT:  [[ADD_0:%.*]] = llvm.add %arg1, [[TRUNC]] : i32
  // CHECK-DAG:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  [[DIV:%.*]] = llvm.udiv [[TRUNC]], [[ONE]] : i32
  // CHECK-NEXT:  [[ADD_1:%.*]] = llvm.add %arg4, [[DIV]] : i32
  // CHECK-DAG:   [[ZERO_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:   [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT:  [[COORD0:%.*]] = llvm.insertelement [[ADD_1]], [[UNDEF]][[[ZERO_1]] : i32] : vector<2xi32>
  // CHECK-NEXT:  [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK-DAG:   [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:   [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG:   [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:   [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], [[BASE_ALIGNED]], [[ADD_0]], %arg2, %arg3, [[COORD1]], [[DEST]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:  llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

// COM: Not supported yet by SPIRV/OCL intrinsics
llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:    [[ONE0:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:    [[PTR:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK:    [[VAL_63:%.*]] = llvm.mlir.constant(-64 : i64) : i64
  // CHECK:    [[VAL_64:%.*]] = llvm.and [[PTR]], [[VAL_63]] : i64
  // CHECK:    [[VAL_65:%.*]] = llvm.inttoptr [[VAL_64]] : i64 to !llvm.ptr<1>
  // CHECK:    [[CL:%.*]] = llvm.mlir.constant(63 : i64) : i64
  // CHECK:    [[AND:%.*]] = llvm.and [[PTR]], [[CL]] : i64
  // CHECK:    [[TRUNC:%.*]] = llvm.trunc [[AND]] : i64 to i32
  // CHECK:    [[ADD:%.*]] = llvm.add %arg1, [[TRUNC]] : i32
  // CHECK:    [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:    [[SHR:%.*]] = llvm.udiv [[TRUNC]], [[ONE]] : i32
  // CHECK:    [[X:%.*]] = llvm.add %arg4, [[SHR]] : i32
  // CHECK:    [[BASE_ALIGNED:%.*]] = llvm.ptrtoint [[VAL_65]] : !llvm.ptr<1> to i64
  // CHECK:    [[BASEWIDTH:%.*]] = llvm.sub [[ADD]], [[ONE0]] : i32
  // CHECK:    [[ELEM_BITS:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:    [[TILE_WIDTH:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:    [[TILE_HEIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:    [[VBLOCKS:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:    [[TRANSPOSE:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    [[VNNI:%.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v8i8([[BASE_ALIGNED]], [[BASEWIDTH]], {{.*}}, [[X]], {{.*}}, [[ELEM_BITS]], [[TILE_WIDTH]], [[TILE_HEIGHT]], [[VBLOCKS]], [[TRANSPOSE]], [[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<16xi8>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<32xi8>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:    %[[ELEM_BITS:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:    %[[TILE_WIDTH:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:    %[[TILE_HEIGHT:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:    %[[VBLOCKS:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK:    %[[TRANSPOSE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    %[[VNNI:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    %[[VAL_68:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v16i8({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[ELEM_BITS]], %[[TILE_WIDTH]], %[[TILE_HEIGHT]], %[[VBLOCKS]], %[[TRANSPOSE]], %[[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi8>
  llvm.return
}

// -----

// COM: This case come from the 06 tutorial of FP8 flash attention.
llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<32xi8>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:    %[[ELEM_BITS:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:    %[[TILE_WIDTH:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK:    %[[TILE_HEIGHT:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK:    %[[VBLOCKS:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK:    %[[TRANSPOSE:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    %[[VNNI:.*]] = llvm.mlir.constant(false) : i1
  // CHECK:    %[[VAL_68:.*]] = llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i16({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[ELEM_BITS]], %[[TILE_WIDTH]], %[[TILE_HEIGHT]], %[[VBLOCKS]], %[[TRANSPOSE]], %[[VNNI]], {{.*}})
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=8, tile_height=16, v_blocks=4, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(4 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<4xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: lvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<1xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=2, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<1xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<64xi16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK-COUNT-2: llvm.mlir.constant(1 : i32) : i32
  // CHECK:         [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:    [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT:    [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT:    [[VBlocks:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT:    llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT:    llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=4, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<8xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<16xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(2 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  // CHECK-NEXT: llvm.load [[DEST]] : !llvm.ptr -> vector<32xi32>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<32xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload_(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(4 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransposeINTELiiiiPU3AS1viiiDv2_iPv([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], {{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, transpose=true, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 4>, #triton_gen.load_cache_control<1, Uncached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1UC_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 4>, #triton_gen.load_cache_control<1, Cached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1UC_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 4>, #triton_gen.load_cache_control<1, Uncached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1C_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 4>, #triton_gen.load_cache_control<1, Cached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1C_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Streaming, 4>, #triton_gen.load_cache_control<1, Uncached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1S_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Streaming, 4>, #triton_gen.load_cache_control<1, Cached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1S_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, InvalidateAfterRead, 4>, #triton_gen.load_cache_control<1, Cached, 4>>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=L1IAR_L3C} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:          llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(
  // CHECK-NOT:        triton_gen.DecorationCacheControlINTEL
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}
