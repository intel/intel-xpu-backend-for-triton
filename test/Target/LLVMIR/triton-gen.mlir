// RUN: triton-translate -triton-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: define spir_kernel void @test_intel_reqd_sub_group_size() !intel_reqd_sub_group_size ![[REQD_SUB_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_intel_reqd_sub_group_size() attributes {triton_gen.intel_reqd_sub_group_size = [32 : i32]} {
  llvm.return
}
// CHECK: define spir_kernel void @test_max_work_group_size() !max_work_group_size ![[MAX_WORK_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_max_work_group_size() attributes {triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  llvm.return
}
// CHECK: define spir_kernel void @test_reqd_work_group_size() !reqd_work_group_size ![[REQD_WORK_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_reqd_work_group_size() attributes {triton_gen.reqd_work_group_size = [128 : i32, 1 : i32, 2 : i32]} {
  llvm.return
}

// CHECK-DAG: ![[REQD_SUB_GROUP_SIZE]] = !{i64 32}
// CHECK-DAG: ![[MAX_WORK_GROUP_SIZE]] = !{i64 128, i64 1, i64 1}
// CHECK-DAG: ![[REQD_WORK_GROUP_SIZE]] = !{i64 128, i64 1, i64 2}

// -----

// CHECK-LABEL: define void @triton_gen.cache_controls(
// CHECK-SAME:                                         ptr %[[#ARG0:]]) {
llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr) {
  %0 = triton_gen.cache_controls %arg0, [#triton_gen.store_cache_control<0, Uncached>, #triton_gen.store_cache_control<1, WriteThrough>, #triton_gen.load_cache_control<0, Cached>, #triton_gen.load_cache_control<1, Uncached>] : !llvm.ptr
  %1 = triton_gen.cache_controls %arg0, [#triton_gen.store_cache_control<0, WriteBack>, #triton_gen.store_cache_control<1, Streaming>, #triton_gen.load_cache_control<0, Streaming>, #triton_gen.load_cache_control<1, InvalidateAfterRead>, #triton_gen.load_cache_control<2, ConstCached>] : !llvm.ptr
  // CHECK: %[[#LOAD:]] = load i32, ptr %[[#ARG0]], align 4, !spirv.DecorationCacheControlINTEL ![[#DECORATION0:]]
  %2 = llvm.load %0 : !llvm.ptr -> i32
  // CHECK: store i32 %[[#LOAD]], ptr %[[#ARG0]], align 4, !spirv.DecorationCacheControlINTEL ![[#DECORATION1:]]
  llvm.store %2, %1 : i32, !llvm.ptr
  llvm.return
}

// CHECK-DAG: ![[#DECORATION0]] = !{![[#CACHECONTROL0:]], ![[#CACHECONTROL1:]], ![[#CACHECONTROL2:]], ![[#CACHECONTROL3:]]}
// CHECK-DAG: ![[#CACHECONTROL0]] = !{i32 6443, i32 0, i32 0, i32 0}
// CHECK-DAG: ![[#CACHECONTROL1]] = !{i32 6443, i32 1, i32 1, i32 0}
// CHECK-DAG: ![[#CACHECONTROL2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK-DAG: ![[#CACHECONTROL3]] = !{i32 6442, i32 1, i32 0, i32 0}
// CHECK-DAG: ![[#DECORATION1]] = !{![[#CACHECONTROL4:]], ![[#CACHECONTROL5:]], ![[#CACHECONTROL6:]], ![[#CACHECONTROL7:]], ![[#CACHECONTROL8:]]}
// CHECK-DAG: ![[#CACHECONTROL4]] = !{i32 6443, i32 0, i32 2, i32 1}
// CHECK-DAG: ![[#CACHECONTROL5]] = !{i32 6443, i32 1, i32 3, i32 1}
// CHECK-DAG: ![[#CACHECONTROL6]] = !{i32 6442, i32 0, i32 2, i32 1}
// CHECK-DAG: ![[#CACHECONTROL7]] = !{i32 6442, i32 1, i32 3, i32 1}
// CHECK-DAG: ![[#CACHECONTROL8]] = !{i32 6442, i32 2, i32 4, i32 1}
