// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z41intel_sub_group_2d_block_write_8b_8r16x1cPU3AS1viiiDv2_iPh(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {no_unwind, will_return}

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  // CHECK:     llvm.func @triton_gen.2Dblockstore(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<8xi8>) {
  // CHECK:       [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-NEXT:  [[STOREVALPTR:%.*]] = llvm.alloca [[C8]] x i8 : (i32) -> !llvm.ptr
  // CHECK-NEXT:  llvm.store %arg6, [[STOREVALPTR]] : vector<8xi8>, !llvm.ptr
  // CHECK-NEXT:  [[PTRTOINT:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
  // CHECK-NEXT:  [[CL:%.*]] = llvm.mlir.constant(63 : i64) : i64
  // CHECK-NEXT:  [[AND:%.*]] = llvm.and [[PTRTOINT]], [[CL]] : i64
  // CHECK-NEXT:  [[TRUNC:%.*]] = llvm.trunc [[AND]] : i64 to i32
  // CHECK-NEXT:  [[ADD_0:%.*]] = llvm.add %arg1, [[TRUNC]] : i32
  // CHECK-DAG:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:  [[DIV:%.*]] = llvm.udiv [[TRUNC]], [[ONE]] : i32
  // CHECK-NEXT:  [[ADD_1:%.*]] = llvm.add %arg4, [[DIV]] : i32
  // CHECK-DAG:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:   [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
  // CHECK-NEXT:  [[COORD0:%.*]] = llvm.insertelement [[ADD_1]], [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
  // CHECK-NEXT:  [[COORD1:%.*]] = llvm.insertelement %arg5, [[COORD0]][[[ONE]] : i32] : vector<2xi32>
  // CHECK-NEXT:  llvm.call spir_funccc @_Z41intel_sub_group_2d_block_write_8b_8r16x1cPU3AS1viiiDv2_iPh(%arg0, [[ADD_0]], %arg2, %arg3, [[COORD1]], [[STOREVALPTR]])
  // CHECK-SAME:       triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.store_cache_control<1, Uncached, 0>>
  // CHECK-SAME:       : (!llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, cache_control=L1UC_L3UC} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // CHECK:      llvm.call spir_funccc @_Z41intel_sub_group_2d_block_write_8b_8r32x1cPU3AS1viiiDv2_iPt(%arg0, %{{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, %{{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}

// -----

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi32>) {
  // CHECK:      llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, %{{.*}}, %arg2, %arg3, {{.*}}, [[DEST:%.*]]) {{.*}} : (!llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
  llvm.return
}
