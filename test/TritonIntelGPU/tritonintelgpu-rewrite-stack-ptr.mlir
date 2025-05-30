// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --tritonintelgpu-rewrite-stack-ptr | FileCheck %s

module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL:   llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  // CHECK-LABEL:   llvm.func spir_kernelcc @kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>)
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    %1 = tt.load %arg1 : !tt.ptr<f32>
    // CHECK: [[LOAD0:%.*]] = llvm.extractelement {{.*}}[{{.*}}]
    // CHECK: [[LOAD1:%.*]] = llvm.extractelement {{.*}}[{{.*}}]
    // CHECK: [[POISON:%.*]] = llvm.mlir.poison : !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @noinline_simple_fn__fp32_fp32_Pfp32__([[LOAD0]], [[LOAD1]], %arg2, [[POISON]], %arg3)
    tt.call @noinline_simple_fn__fp32_fp32_Pfp32__(%0, %1, %arg2) : (f32, f32, !tt.ptr<f32>) -> ()
    tt.return
  }
  // CHECK:   llvm.func internal spir_funccc @noinline_simple_fn__fp32_fp32_Pfp32__(%arg0: f32, %arg1: f32, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<3>, %arg4: !llvm.ptr<1>)
  tt.func private @noinline_simple_fn__fp32_fp32_Pfp32__(%arg0: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg1: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg2: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64})  attributes {noinline = true} {
    %0 = arith.addf %arg0, %arg1 fastmath<fast> : f32
    tt.store %arg2, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 1280 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL:   llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  // CHECK-LABEL:   llvm.func spir_kernelcc @kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<3>)
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    %1 = tt.load %arg1 : !tt.ptr<f32>
    // CHECK: [[LOAD0:%.*]] = llvm.extractelement {{.*}}[{{.*}}]
    // CHECK: [[LOAD1:%.*]] = llvm.extractelement {{.*}}[{{.*}}]
    // CHECK: llvm.call spir_funccc @noinline_shared_fn__fp32_fp32_Pfp32__([[LOAD0]], [[LOAD1]], %arg2, %arg4, %arg3)
    tt.call @noinline_shared_fn__fp32_fp32_Pfp32__(%0, %1, %arg2) {allocation.offset = 0 : i32} : (f32, f32, !tt.ptr<f32>) -> ()
    tt.return
  }
  // CHECK: llvm.func internal spir_funccc @noinline_shared_fn__fp32_fp32_Pfp32__(%arg0: f32, %arg1: f32, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<3>, %arg4: !llvm.ptr<1>)
  // CHECK: llvm.getelementptr %arg3[{{.*}}]
  tt.func private @noinline_shared_fn__fp32_fp32_Pfp32__(%arg0: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg1: f32 {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 1 : i64}, %arg2: !tt.ptr<f32> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64}) attributes {noinline = true} {
    %cst = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst : tensor<16x1xi32, #blocked>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %5 = tt.broadcast %2 : tensor<16x1xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %6 = tt.broadcast %4 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %7 = arith.addi %5, %6 : tensor<16x16xi32, #blocked>
    %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
    %11 = ttg.local_alloc %10 {allocation.offset = 0 : i32} : (tensor<16x16xf32, #blocked>) -> !ttg.memdesc<16x16xf32, #shared, #smem>
    %12 = tt.splat %arg0 : f32 -> tensor<16x16xf32, #mma>
    %13 = ttg.local_load %11 : !ttg.memdesc<16x16xf32, #shared, #smem> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %14 = ttg.local_load %11 : !ttg.memdesc<16x16xf32, #shared, #smem> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %15 = tt.dot %13, %14, %12, inputPrecision = tf32 : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
    %16 = tt.splat %arg1 : f32 -> tensor<16x16xf32, #mma>
    %17 = arith.addf %15, %16 fastmath<fast> : tensor<16x16xf32, #mma>
    %18 = ttg.convert_layout %17 {allocation.offset = 0 : i32} : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.store %9, %18 : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
