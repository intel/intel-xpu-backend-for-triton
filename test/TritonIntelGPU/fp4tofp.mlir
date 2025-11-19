// RUN: triton-opt %s -split-input-file --triton-annotate-module='support-f4-conversion' --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --check-prefixes=CHECK-SPIRV
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm  -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir16", ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fp4_to_bf16_kernel(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<bf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_tensor_ptr %arg0, [%c16_i64], [%c1_i64], [%1] {order = array<i32: 0>, tt.divisibility = dense<16> : tensor<1xi32>} : <tensor<16xi8, #blocked>>
    %3 = tt.load %2 : !tt.ptr<tensor<16xi8, #blocked>>
    %4 = ttg.fp4_to_fp %3 {axis = 0 : i32} : tensor<16xi8, #blocked> -> tensor<32xbf16, #blocked1>
    %5 = arith.muli %0, %c32_i32 : i32
    %6 = tt.make_tensor_ptr %arg1, [%c32_i64], [%c1_i64], [%5] {order = array<i32: 0>, tt.divisibility = dense<16> : tensor<1xi32>} : <tensor<32xbf16, #blocked1>>
    tt.store %6, %4 : !tt.ptr<tensor<32xbf16, #blocked1>>
    tt.return
  }
}

// CHECK-SPIRV-COUNT-2: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv16_i

// CHECK-DAG: [[C4V:%.+]] = llvm.mlir.constant(dense<4> : vector<4xi32>) : vector<4xi32>
// CHECK-DAG: [[C15V:%.+]] = llvm.mlir.constant(dense<252645135> : vector<4xi32>) : vector<4xi32>
// CHECK-DAG: [[TABLE:%.+]] = llvm.mlir.constant(dense<[0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, -0.000000e+00, -5.000000e-01, -1.000000e+00, -1.500000e+00, -2.000000e+00, -3.000000e+00, -4.000000e+00, -6.000000e+00]> : vector<16xbf16>) : vector<16xbf16>
// CHECK-DAG: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: [[I32V:%.+]] = llvm.load {{.+}} : !llvm.ptr<1> -> vector<4xi32>
// CHECK: [[IDXV0I32:%.+]] = llvm.and [[I32V]], [[C15V]] : vector<4xi32>
// CHECK: [[LSHR:%.+]] = llvm.lshr [[I32V]], [[C4V]] : vector<4xi32>
// CHECK: [[IDXV1I32:%.+]] = llvm.and [[LSHR]], [[C15V]] : vector<4xi32>
// CHECK: [[IDX0I32:%.+]] = llvm.extractelement [[IDXV0I32]][[[C3]] : i32] : vector<4xi32>
// CHECK: [[IDX1I32:%.+]] = llvm.extractelement [[IDXV1I32]][[[C3]] : i32] : vector<4xi32>
// CHECK: [[IDXV0I8:%.+]] = llvm.bitcast [[IDX0I32]] : i32 to vector<4xi8>
// CHECK: [[IDXV1I8:%.+]] = llvm.bitcast [[IDX1I32]] : i32 to vector<4xi8>
// CHECK: [[IDX0I8:%.+]] = llvm.extractelement [[IDXV0I8]][[[C3]] : i32] : vector<4xi8>
// CHECK: [[V0:%.+]] = llvm.extractelement [[TABLE]][[[IDX0I8]] : i8] : vector<16xbf16>
// CHECK: [[IDX1I8:%.+]] = llvm.extractelement [[IDXV1I8]][[[C3]] : i32] : vector<4xi8>
// CHECK: [[V1:%.+]] = llvm.extractelement [[TABLE]][[[IDX1I8]] : i8] : vector<16xbf16>

// -----

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir16", ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fp4_to_bf16_kernel(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<bf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_tensor_ptr %arg0, [%c16_i64], [%c1_i64], [%1] {order = array<i32: 0>, tt.divisibility = dense<4> : tensor<1xi32>} : <tensor<16xi8, #blocked>>
    %3 = tt.load %2 : !tt.ptr<tensor<16xi8, #blocked>>
    %4 = ttg.fp4_to_fp %3 {axis = 0 : i32} : tensor<16xi8, #blocked> -> tensor<32xbf16, #blocked1>
    %5 = arith.muli %0, %c32_i32 : i32
    %6 = tt.make_tensor_ptr %arg1, [%c32_i64], [%c1_i64], [%5] {order = array<i32: 0>, tt.divisibility = dense<4> : tensor<1xi32>} : <tensor<32xbf16, #blocked1>>
    tt.store %6, %4 : !tt.ptr<tensor<32xbf16, #blocked1>>
    tt.return
  }
}

// CHECK-SPIRV-COUNT-2: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv16_i

// CHECK-DAG: [[C4V:%.+]] = llvm.mlir.constant(dense<4> : vector<4xi8>) : vector<4xi8>
// CHECK-DAG: [[C15V:%.+]] = llvm.mlir.constant(dense<15> : vector<4xi8>) : vector<4xi8>
// CHECK-DAG: [[TABLE:%.+]] = llvm.mlir.constant(dense<[0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, -0.000000e+00, -5.000000e-01, -1.000000e+00, -1.500000e+00, -2.000000e+00, -3.000000e+00, -4.000000e+00, -6.000000e+00]> : vector<16xbf16>) : vector<16xbf16>
// CHECK-DAG: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: [[I32:%.+]] = llvm.load {{.+}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
// CHECK: [[IDXVI8:%.+]] = llvm.bitcast [[I32]] : i32 to vector<4xi8>
// CHECK: [[IDXV0I8:%.+]] = llvm.and [[IDXVI8]], [[C15V]] : vector<4xi8>
// CHECK: [[IDXV1I8:%.+]] = llvm.lshr [[IDXVI8]], [[C4V]] : vector<4xi8>
// CHECK: [[IDX0I8:%.+]] = llvm.extractelement [[IDXV0I8]][[[C3]] : i32] : vector<4xi8>
// CHECK: [[V0:%.+]] = llvm.extractelement [[TABLE]][[[IDX0I8]] : i8] : vector<16xbf16>
// CHECK: [[IDX1I8:%.+]] = llvm.extractelement [[IDXV1I8]][[[C3]] : i32] : vector<4xi8>
// CHECK: [[V1:%.+]] = llvm.extractelement [[TABLE]][[[IDX1I8]] : i8] : vector<16xbf16>

// -----

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir16", ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fp4_to_bf16_kernel(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<bf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_tensor_ptr %arg0, [%c16_i64], [%c1_i64], [%1] {order = array<i32: 0>, tt.divisibility = dense<1> : tensor<1xi32>} : <tensor<16xi8, #blocked>>
    %3 = tt.load %2 : !tt.ptr<tensor<16xi8, #blocked>>
    %4 = ttg.fp4_to_fp %3 {axis = 0 : i32} : tensor<16xi8, #blocked> -> tensor<32xbf16, #blocked1>
    %5 = arith.muli %0, %c32_i32 : i32
    %6 = tt.make_tensor_ptr %arg1, [%c32_i64], [%c1_i64], [%5] {order = array<i32: 0>, tt.divisibility = dense<1> : tensor<1xi32>} : <tensor<32xbf16, #blocked1>>
    tt.store %6, %4 : !tt.ptr<tensor<32xbf16, #blocked1>>
    tt.return
  }
}

// CHECK-SPIRV-COUNT-2: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToBF16INTELDv16_i

// CHECK-DAG: [[C4:%.+]] = llvm.mlir.constant(4 : i8) : i8
// CHECK-DAG: [[C15:%.+]] = llvm.mlir.constant(15 : i8) : i8
// CHECK-DAG: [[TABLE:%.+]] = llvm.mlir.constant(dense<[0.000000e+00, 5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, -0.000000e+00, -5.000000e-01, -1.000000e+00, -1.500000e+00, -2.000000e+00, -3.000000e+00, -4.000000e+00, -6.000000e+00]> : vector<16xbf16>) : vector<16xbf16>
// CHECK-DAG: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG: [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-COUNT-16: [[I8:%.+]] = llvm.load {{.+}} {alignment = 1 : i64} : !llvm.ptr<1> -> i8
// CHECK: [[I8V:%.+]] = llvm.bitcast [[I8]] : i8 to vector<1xi8>
// CHECK: [[I8:%.+]] = llvm.extractelement [[I8V]][[[C0]] : i32] : vector<1xi8>
// CHECK: [[IDX0I8:%.+]] = llvm.and [[I8]], [[C15]] : i8
// CHECK: [[IDX1I8:%.+]] = llvm.lshr [[I8]], [[C4]] : i8
// CHECK: [[V0:%.+]] = llvm.extractelement [[TABLE]][[[IDX0I8]] : i8] : vector<16xbf16>
// CHECK: [[V1:%.+]] = llvm.extractelement [[TABLE]][[[IDX1I8]] : i8] : vector<16xbf16>

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<4xi8, #blocked>) -> tensor<8xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<4xi8, #blocked> -> tensor<8xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv8_i
    tt.return %dst : tensor<8xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<2xi8, #blocked>) -> tensor<4xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<2xi8, #blocked> -> tensor<4xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv4_i
    tt.return %dst : tensor<4xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<1xi8, #blocked>) -> tensor<2xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<1xi8, #blocked> -> tensor<2xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z38__builtin_spirv_ConvertE2M1ToFP16INTELDv2_i
    tt.return %dst : tensor<2xf16, #blocked1>
  }
}
