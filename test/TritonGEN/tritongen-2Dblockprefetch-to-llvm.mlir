// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:     llvm.func @triton_gen.2Dblockprefetch(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-DAG:  [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT: [[COORD0:%.*]] = llvm.insertelement %arg4, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT: [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, [[COORD1]])
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 4>, #triton_gen.load_cache_control<1, Uncached, 4>>
  // CHECK-SAME:       : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=L1UC_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=16, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=16, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=32, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=16, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=32, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=2, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.mlir.constant(1 : i32) : i32
  // CHECK:      [[ElemSize:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT: [[TileWidth:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-NEXT: [[TileHeight:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-NEXT: [[VBlocks:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i([[ElemSize]], [[TileWidth]], [[TileHeight]], [[VBlocks]], %arg0, %arg1, %arg2, %arg3, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>) -> ()
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=32, v_blocks=4, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
