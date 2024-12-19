// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Basic 16x16 SIMD reduction.

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test_single(
// CHECK-SAME:                                         %[[VAL_0:.*]]: !llvm.struct
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.poison : vector<16xf32>
// COM: Check we insert all tensor elements in a vector:
// CHECK-COUNT-16:  llvm.insertelement
// CHECK:           %[[VAL_50:.*]] = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "{\0A.decl temp_result v_type=G type=f num_elts=128 align=wordx32\0Aadd (M1_NM, 16) temp_result(0, 0)<1> $1(0, 0)<16;8,1> $1(0, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(1, 0)<1> $1(2, 0)<16;8,1> $1(2, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(2, 0)<1> $1(4, 0)<16;8,1> $1(4, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(3, 0)<1> $1(6, 0)<16;8,1> $1(6, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(4, 0)<1> $1(8, 0)<16;8,1> $1(8, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(5, 0)<1> $1(10, 0)<16;8,1> $1(10, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(6, 0)<1> $1(12, 0)<16;8,1> $1(12, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(7, 0)<1> $1(14, 0)<16;8,1> $1(14, 8)<16;8,1>\0Aadd (M1_NM, 16) temp_result(0, 0)<1> temp_result(0, 0)<8;4,1> temp_result(0, 4)<8;4,1>\0Aadd (M1_NM, 16) temp_result(1, 0)<1> temp_result(2, 0)<8;4,1> temp_result(2, 4)<8;4,1>\0Aadd (M1_NM, 16) temp_result(2, 0)<1> temp_result(4, 0)<8;4,1> temp_result(4, 4)<8;4,1>\0Aadd (M1_NM, 16) temp_result(3, 0)<1> temp_result(6, 0)<8;4,1> temp_result(6, 4)<8;4,1>\0Aadd (M1_NM, 16) temp_result(0, 0)<1> temp_result(0, 0)<4;2,1> temp_result(0, 2)<4;2,1>\0Aadd (M1_NM, 16) temp_result(1, 0)<1> temp_result(2, 0)<4;2,1> temp_result(2, 2)<4;2,1>\0Aadd (M1_NM, 16) $0(0, 0)<1> temp_result(0, 0)<2;1,0> temp_result(0, 1)<2;1,0>\0A}", "=rw,rw" %{{.*}} : (vector<16xf32>) -> f32
// COM: Check we obtain a single result, i.e., the SIMD reduction minimizes register usage.
// CHECK:           %[[VAL_51:.*]] = llvm.mlir.undef : !llvm.struct<(f32)>
// CHECK:           %[[VAL_52:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_51]][0] : !llvm.struct<(f32)>
// CHECK:           llvm.return %[[VAL_52]] : !llvm.struct<(f32)>
// CHECK:         }
  tt.func @test_single(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked1> {
    %0 = triton_intel_gpu.simd_reduce add %arg0 axis = 0 : tensor<16x16xf32, #blocked> -> tensor<16xf32, #blocked1>
    tt.return %0 : tensor<16xf32, #blocked1>
  }
}
