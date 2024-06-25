// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize  -convert-triton-to-tritongpu=target="xpu:DEVICE_ARCH.PVC threads-per-warp=16" -tritongpu-remove-layout-conversions  --tritonintelgpu-accelerate-matmul --tritonintelgpu-remove-layout-conversions --tritonintelgpu-rewrite-tensor-pointer --intel-decompose-unsupported-conversions --convert-scf-to-cf --convert-index-to-llvm --convert-arith-to-llvm --canonicalize --cse --symbol-dce --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh(vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16> attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {passthrough = ["nounwind"]}
// CHECK-DAG: llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {passthrough = ["nounwind"]}
module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
tt.func @test_addptr_dot_op(%arg0 : !tt.ptr<f16>, %arg1: i64) {
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
  %c1_i1 = arith.constant 1 : i1
  %10 = tt.splat %c1_i1 : i1 -> tensor<1x64xi1>
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
  %1 = tt.make_range {start = 128 : i32, end = 256 : i32} : tensor<128xi32>
  %21 = tt.expand_dims %1 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %2 = tt.addptr %0, %21 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
  // CHECK-COUNT-4: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-COUNT-4: llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj({{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
  // CHECK-COUNT-8: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh({{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16>
  %3 = tt.load %2 : tensor<1x128x!tt.ptr<f16>>
  %7 = tt.broadcast %3 : tensor<1x128xf16> -> tensor<64x128xf16>
  %8 = tt.broadcast %3 : tensor<1x128xf16> -> tensor<128x64xf16>	
  %78 = tt.dot %7, %8, %cst_0 : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
  %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>	   
  %5 = tt.splat %arg1 : i64 -> tensor<64x64xi64>
  %6 = tt.addptr %4, %5 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi64>
  %11 = tt.broadcast %10 : tensor<1x64xi1> -> tensor<64x64xi1>
  tt.store %6, %78, %11 : tensor<64x64x!tt.ptr<f16>>
  tt.return
}
}
