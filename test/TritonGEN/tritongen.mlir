// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @triton_gen_special_regs() -> i32 {
  // CHECK-LABEL: triton_gen_special_regs
  // CHECK: triton_gen.workitem.id.x : i32
  %0 = triton_gen.workitem.id.x : i32
  // CHECK: triton_gen.workitem.id.y : i32
  %1 = triton_gen.workitem.id.y : i32
  // CHECK: triton_gen.workitem.id.z : i32
  %2 = triton_gen.workitem.id.z : i32
  // CHECK: triton_gen.workgroup.id.x : i32
  %3 = triton_gen.workgroup.id.x : i32
  // CHECK: triton_gen.workgroup.id.y : i32
  %4 = triton_gen.workgroup.id.y : i32
  // CHECK: triton_gen.workgroup.id.z : i32
  %5 = triton_gen.workgroup.id.z : i32
  // CHECK: triton_gen.workgroup.dim.x : i32
  %6 = triton_gen.workgroup.dim.x : i32
  // CHECK: triton_gen.workgroup.dim.y : i32
  %7 = triton_gen.workgroup.dim.y : i32
  // CHECK: triton_gen.workgroup.dim.z : i32
  %8 = triton_gen.workgroup.dim.z : i32
  // CHECK: triton_gen.grid.dim.x : i32
  %9 = triton_gen.grid.dim.x : i32
  // CHECK: triton_gen.grid.dim.y : i32
  %10 = triton_gen.grid.dim.y : i32
  // CHECK: triton_gen.grid.dim.z : i32
  %11 = triton_gen.grid.dim.z : i32
  // CHECK: triton_gen.subgroup.id : i32
  %12 = triton_gen.subgroup.id : i32
  llvm.return %0 : i32
}

llvm.func @triton_gen.barrier() {
  // CHECK-LABEL: triton_gen.barrier
  // CHECK: triton_gen.barrier {mem_fence = Local}
  triton_gen.barrier {mem_fence=Local}
  llvm.return
}

llvm.func @triton_gen.split_barrier_signal() {
  // CHECK-LABEL: triton_gen.split_barrier_signal
  // CHECK: triton_gen.split_barrier_signal {mem_fence = None, mem_scope = WorkGroup}
  triton_gen.split_barrier_signal {mem_fence=None, mem_scope=WorkGroup}
  llvm.return
}

llvm.func @triton_gen.split_barrier_wait() {
  // CHECK-LABEL: triton_gen.split_barrier_wait
  // CHECK: triton_gen.split_barrier_wait {mem_fence = Local, mem_scope = SubGroup}
  triton_gen.split_barrier_wait {mem_fence=Local, mem_scope=SubGroup}
  llvm.return
}

llvm.func @triton_gen.named_barrier_signal(%barrier_id : i32, %thread_group_count : i32) {
  // CHECK-LABEL: triton_gen.named_barrier_signal(%arg0: i32, %arg1: i32) {
  // CHECK-NEXT: triton_gen.named_barrier_signal %arg0, %arg1 : (i32, i32)
  triton_gen.named_barrier_signal %barrier_id, %thread_group_count : (i32, i32)
  llvm.return
}

llvm.func @triton_gen.named_barrier_wait(%barrier_id : i32) {
  // CHECK-LABEL: triton_gen.named_barrier_wait(%arg0: i32) {
  // CHECK-NEXT: triton_gen.named_barrier_wait %arg0 : i32
  triton_gen.named_barrier_wait %barrier_id : i32
  llvm.return
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 32>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // CHECK-LABEL: triton_gen.sub_group_reduce
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.sub_group_reduce add %0 {size = 16} : i32
    %1 = triton_gen.sub_group_reduce add %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce mul %0 {size = 16} : i32
    %2 = triton_gen.sub_group_reduce mul %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce min %0 {size = 16} : i32
    %3 = triton_gen.sub_group_reduce min %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce max %0 {size = 16} : i32
    %4 = triton_gen.sub_group_reduce max %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce and %0 {size = 16} : i32
    %5 = triton_gen.sub_group_reduce and %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce or %0 {size = 16} : i32
    %6 = triton_gen.sub_group_reduce or %0 {size = 16} : i32
    // CHECK: triton_gen.sub_group_reduce xor %0 {size = 16} : i32
    %7 = triton_gen.sub_group_reduce xor %0 {size = 16} : i32
    llvm.return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 32>>
} {
  llvm.func @triton_gen.sub_group_scan() {
    // CHECK-LABEL: triton_gen.sub_group_scan
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.sub_group_scan add %0 {kind = exclusive} : i32
    %1 = triton_gen.sub_group_scan add %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan mul %0 {kind = exclusive} : i32
    %2 = triton_gen.sub_group_scan mul %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan min %0 {kind = exclusive} : i32
    %3 = triton_gen.sub_group_scan min %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan max %0 {kind = exclusive} : i32
    %4 = triton_gen.sub_group_scan max %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan and %0 {kind = exclusive} : i32
    %5 = triton_gen.sub_group_scan and %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan or %0 {kind = exclusive} : i32
    %6 = triton_gen.sub_group_scan or %0 {kind = exclusive} : i32
    // CHECK: triton_gen.sub_group_scan xor %0 {kind = exclusive} : i32
    %7 = triton_gen.sub_group_scan xor %0 {kind = exclusive} : i32

    // CHECK: triton_gen.sub_group_scan add %0 {kind = inclusive} : i32
    %8 = triton_gen.sub_group_scan add %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan mul %0 {kind = inclusive} : i32
    %9 = triton_gen.sub_group_scan mul %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan min %0 {kind = inclusive} : i32
    %10 = triton_gen.sub_group_scan min %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan max %0 {kind = inclusive} : i32
    %11 = triton_gen.sub_group_scan max %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan and %0 {kind = inclusive} : i32
    %12 = triton_gen.sub_group_scan and %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan or %0 {kind = inclusive} : i32
    %13 = triton_gen.sub_group_scan or %0 {kind = inclusive} : i32
    // CHECK: triton_gen.sub_group_scan xor %0 {kind = inclusive} : i32
    %14 = triton_gen.sub_group_scan xor %0 {kind = inclusive} : i32

    llvm.return
  }
}

// -----

llvm.func @triton_gen.sub_group_shuffle() {
  // CHECK-LABEL: triton_gen.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = triton_gen.sub_group_shuffle xor %0, %0 : i32
  %1 = triton_gen.sub_group_shuffle xor %0, %0 : i32
  // CHECK: %2 = triton_gen.sub_group_shuffle up %0, %0 : i32
  %2 = triton_gen.sub_group_shuffle up %0, %0 : i32
  // CHECK: %3 = triton_gen.sub_group_shuffle down %0, %0 : i32
  %3 = triton_gen.sub_group_shuffle down %0, %0 : i32
  // CHECK: %4 = triton_gen.sub_group_shuffle idx %0, %0 : i32
  %4 = triton_gen.sub_group_shuffle idx %0, %0 : i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = triton_gen.sub_group_shuffle xor %5, %0 : i8
  %6 = triton_gen.sub_group_shuffle xor %5, %0 : i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = triton_gen.sub_group_shuffle xor %7, %0 : i16
  %8 = triton_gen.sub_group_shuffle xor %7, %0 : i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = triton_gen.sub_group_shuffle xor %9, %0 : i64
  %10 = triton_gen.sub_group_shuffle xor %9, %0 : i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = triton_gen.sub_group_shuffle xor %11, %0 : f16
  %12 = triton_gen.sub_group_shuffle xor %11, %0 : f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = triton_gen.sub_group_shuffle xor %13, %0 : f32
  %14 = triton_gen.sub_group_shuffle xor %13, %0 : f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = triton_gen.sub_group_shuffle xor %15, %0 : f64
  %16 = triton_gen.sub_group_shuffle xor %15, %0 : f64
  llvm.return
}

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:      llvm.func @triton_gen.dpas(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT:   %0 = triton_gen.dpas %arg0, %arg1, %arg2 {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)

llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr) {
  // CHECK: llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr)
  // CHECK-NEXT: %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
  %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
  // CHECK-NEXT: llvm.store %0, %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, WriteBack, 1>, #triton_gen.store_cache_control<1, Streaming, 1>>} : i32, !llvm.ptr
  llvm.store %0, %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, WriteBack, 1>, #triton_gen.store_cache_control<1, Streaming, 1>>} : i32, !llvm.ptr
  // CHECK-NEXT: llvm.call @foo(%arg0, %arg0) {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 1>>} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @foo(%arg0, %arg0) {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 1>>} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:   %0 = triton_gen.2Dblockload %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<16xf32>) {
  // CHECK:      llvm.func @triton_gen.2Dblockstore(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<16xf32>) {
  // CHECK-NEXT:   triton_gen.2Dblockstore %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<16xf32>)
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<16xf32>)
  llvm.return
}

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockprefetch(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:    triton_gen.2Dblockprefetch %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

llvm.func @triton_gen.simdblockread(%ptr : !llvm.ptr) {
  // CHECK:      llvm.func @triton_gen.simdblockread(%arg0: !llvm.ptr) {
  // CHECK-NEXT:   triton_gen.simdblockread %arg0 : (!llvm.ptr) -> vector<64xi16>
  triton_gen.simdblockread %ptr : (!llvm.ptr) -> vector<64xi16>
  llvm.return
}

llvm.func @triton_gen.simdblockwrite(%ptr : !llvm.ptr, %val : vector<64xi16>) {
  // CHECK:      llvm.func @triton_gen.simdblockwrite(%arg0: !llvm.ptr, %arg1: vector<64xi16>) {
  // CHECK-NEXT:    triton_gen.simdblockwrite %arg0, %arg1 : (!llvm.ptr, vector<64xi16>)
  triton_gen.simdblockwrite %ptr, %val : (!llvm.ptr, vector<64xi16>)
  llvm.return
}
