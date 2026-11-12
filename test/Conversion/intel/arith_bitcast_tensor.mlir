// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Regression test: arith.bitcast on a tensor whose per-thread register count > 1
// must be lowered element-wise (one scalar llvm.bitcast per element).
//
// Root cause: mlir::arith::populateArithToLLVMConversionPatterns registers
// BitcastOpLowering (benefit=1) which calls
//   rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, llvmDstTy, adaptor.getIn())
// When sizePerThread > 1 the tensor is represented as an LLVM struct, so
// llvmDstTy is also a struct and the result is:
//   llvm.bitcast !llvm.struct<(bf16,bf16)> to !llvm.struct<(i16,i16)>
// LLVM rejects this with:
//   error: 'llvm.bitcast' op operand #0 must be LLVM-compatible non-aggregate type
//
// The fix (ArithBitcastOpConversion, benefit=patternBenefitPrioritizeOverLLVMConversions)
// overrides the upstream pattern and uses ElementwiseOpConversionBase to unpack
// the struct, scalar-bitcast each element, and repack—producing:
//   llvm.bitcast %elem : bf16 to i16   (repeated per element)

// Two elements per thread → struct<(bf16, bf16)>
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttig.support_2d_block_io", "ttig.support_subgroup_matrix_multiply_accumulate", "ttig.support_bfloat16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @bitcast_bf16_to_i16
  // Input tensor lowers to struct<(bf16, bf16)>. The fix extracts each element,
  // applies a scalar bitcast (valid in LLVM), then repacks.
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(bf16, bf16)>
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(bf16, bf16)>
  // CHECK: llvm.bitcast {{.*}} : bf16 to i16
  // CHECK: llvm.bitcast {{.*}} : bf16 to i16
  // CHECK: llvm.insertvalue {{.*}} : !llvm.struct<(i16, i16)>
  // CHECK: llvm.insertvalue {{.*}} : !llvm.struct<(i16, i16)>
  // COM: No struct-level bitcast — that would fail LLVM verification.
  // CHECK-NOT: llvm.bitcast {{.*}} : !llvm.struct
  tt.func @bitcast_bf16_to_i16(%arg0: tensor<256xbf16, #blocked>) -> tensor<256xi16, #blocked> {
    %0 = arith.bitcast %arg0 : tensor<256xbf16, #blocked> to tensor<256xi16, #blocked>
    tt.return %0 : tensor<256xi16, #blocked>
  }
}

// -----

// Also verify the reverse direction (i16 → bf16) works correctly.
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttig.support_2d_block_io", "ttig.support_subgroup_matrix_multiply_accumulate", "ttig.support_bfloat16_conversion", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @bitcast_i16_to_bf16
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(i16, i16)>
  // CHECK: llvm.extractvalue {{.*}} : !llvm.struct<(i16, i16)>
  // CHECK: llvm.bitcast {{.*}} : i16 to bf16
  // CHECK: llvm.bitcast {{.*}} : i16 to bf16
  // CHECK: llvm.insertvalue {{.*}} : !llvm.struct<(bf16, bf16)>
  // CHECK: llvm.insertvalue {{.*}} : !llvm.struct<(bf16, bf16)>
  // CHECK-NOT: llvm.bitcast {{.*}} : !llvm.struct
  tt.func @bitcast_i16_to_bf16(%arg0: tensor<256xi16, #blocked>) -> tensor<256xbf16, #blocked> {
    %0 = arith.bitcast %arg0 : tensor<256xi16, #blocked> to tensor<256xbf16, #blocked>
    tt.return %0 : tensor<256xbf16, #blocked>
  }
}
