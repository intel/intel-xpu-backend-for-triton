// RUN: triton-translate -triton-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: define spir_kernel void @test_max_work_group_size() !max_work_group_size ![[MAX_WORK_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_max_work_group_size() attributes {triton_gen.max_work_group_size = array<i32: 128, 1, 1>} {
  llvm.return
}

// CHECK-DAG: ![[MAX_WORK_GROUP_SIZE]] = !{i64 128, i64 1, i64 1}

// -----

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)

// CHECK-LABEL: define void @triton_gen.cache_controls(
// CHECK-SAME:                                         ptr %[[#ARG0:]]) {
llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr) {
  // CHECK: %[[#LOAD:]] = load i32, ptr %[[#ARG0]], align 4, !spirv.DecorationCacheControlINTEL ![[#DECORATION0:]]
  %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
  // CHECK: store i32 %[[#LOAD]], ptr %[[#ARG0]], align 4, !spirv.DecorationCacheControlINTEL ![[#DECORATION1:]]
  llvm.store %0, %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, WriteBack, 1>, #triton_gen.store_cache_control<1, Streaming, 1>>} : i32, !llvm.ptr
  // CHECK: call void @foo(ptr %[[#ARG0]], ptr %[[#ARG0]]), !spirv.DecorationCacheControlINTEL ![[#DECORATION2:]]
  llvm.call @foo(%arg0, %arg0) {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 1>>} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

// CHECK-DAG: ![[#DECORATION0]] = !{![[#CACHECONTROL0:]], ![[#CACHECONTROL1:]]}
// CHECK-DAG: ![[#DECORATION1]] = !{![[#CACHECONTROL2:]], ![[#CACHECONTROL3:]]}
// CHECK-DAG: ![[#DECORATION2]] = !{![[#CACHECONTROL4:]], ![[#CACHECONTROL5:]]}
// CHECK-DAG: ![[#CACHECONTROL0]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK-DAG: ![[#CACHECONTROL1]] = !{i32 6442, i32 1, i32 0, i32 0}
// CHECK-DAG: ![[#CACHECONTROL2]] = !{i32 6443, i32 0, i32 2, i32 1}
// CHECK-DAG: ![[#CACHECONTROL3]] = !{i32 6443, i32 1, i32 3, i32 1}
// CHECK-DAG: ![[#CACHECONTROL4]] = !{i32 6443, i32 0, i32 0, i32 0}
// CHECK-DAG: ![[#CACHECONTROL5]] = !{i32 6442, i32 0, i32 1, i32 1}
