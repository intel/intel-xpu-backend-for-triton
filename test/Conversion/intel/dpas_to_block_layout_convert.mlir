// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 67584 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>)
// CHECK-SAME:                                          attributes {intel_reqd_sub_group_size = 16 : i32, noinline = false, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:       %[[SMEM:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>

    // COM: Flat SLM path: the DPAS layout values are stored to SLM in vectorized way.
    // COM: 8 stores of vector<8xf16> are generated. 128*256 elems / (32 warps * 16 lanes * 8 elems_per_store) = 8
    // CHECK-COUNT-8:   llvm.store {{.*}} : vector<8xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj({{.*}}) {convergent, no_unwind, will_return} : (i32) -> ()

    // COM: The blocked layout values are loaded from SLM as scalar (vector<1xf16>) loads.
    // COM: 64 loads are generated. 128*256 / (32 warps * 16 lanes * 1 elem_per_load) = 64
    // CHECK-COUNT-64:  llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>

    %93 = ttg.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}


// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [2, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.shared = 67584 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @convert_dpas(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr<1>)
// CHECK-SAME:                                          attributes {intel_reqd_sub_group_size = 16 : i32, noinline = false, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @convert_dpas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #mma>

    // CHECK-DAG:       %[[SMEM:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>

    // COM: Flat SLM path: the DPAS layout values are stored to SLM in vectorized way.
    // COM: 8 stores of vector<8xf16> are generated (repCluster [2,2] variant).
    // CHECK-COUNT-8:   llvm.store {{.*}} : vector<8xf16>, !llvm.ptr<3>
    // CHECK:           llvm.call spir_funccc @_Z7barrierj({{.*}}) {convergent, no_unwind, will_return} : (i32) -> ()

    // COM: The blocked layout values are loaded from SLM as scalar (vector<1xf16>) loads.
    // COM: 64 loads are generated.
    // CHECK-COUNT-64:  llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xf16>
    %93 = ttg.convert_layout %cst {allocation.offset = 0 : i32} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
    %80 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %83 = tt.broadcast %80 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.store %83, %93 : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
