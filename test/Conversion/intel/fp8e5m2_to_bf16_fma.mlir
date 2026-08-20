// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Non-LTS path — fast fmul-based converter (requires ttig.support_bfloat16_arithmetic)
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_arithmetic, ttig.target_arch = "spir64" } {
  // CHECK-LABEL: @convert_fp8e5m2_to_bf16_fmul
  tt.func public @convert_fp8e5m2_to_bf16_fmul(%src: tensor<16xf8E5M2, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E5M2, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: %[[MUL:.*]] = llvm.mlir.constant(5.192300e+33 : bf16) : bf16
    // CHECK-DAG: %[[NOSIGN:.*]] = llvm.mlir.constant(2147450879 : i32) : i32
    // CHECK-DAG: %[[SIGNMASK:.*]] = llvm.mlir.constant(-2147450880 : i32) : i32
    // CHECK-DAG: %[[EXPMASK:.*]] = llvm.mlir.constant(124 : i8) : i8
    // CHECK-DAG: %[[MANTMASK:.*]] = llvm.mlir.constant(3 : i8) : i8
    // CHECK-DAG: %[[NAN:.*]] = llvm.mlir.constant(0x7FC0 : bf16) : bf16
    // CHECK-DAG: %[[INFBITS:.*]] = llvm.mlir.constant(32640 : i16) : i16
    // CHECK: llvm.and {{.*}}, %[[NOSIGN]] : i32
    // CHECK: llvm.lshr {{.*}} : i32
    // CHECK: llvm.and {{.*}}, %[[SIGNMASK]] : i32
    // CHECK: llvm.or {{.*}} : i32
    // CHECK: llvm.bitcast {{.*}} : i32 to vector<2xbf16>
    // CHECK: llvm.fmul {{.*}} : vector<2xbf16>
    // COM: NaN/Inf fixup for the fmul path: FP8E5M2 reserves exponent all-ones
    // COM: for Inf (mantissa 0) and NaN (mantissa != 0), so byte & 0x7C == 0x7C
    // COM: selects a special bf16 value in place of the finite fmul result. The
    // COM: Inf/NaN choice, the sign-preserving OR chain and both select operand
    // COM: orders are pinned so an operand swap or dropped fixup cannot pass.
    // CHECK: %[[INF:.*]] = llvm.bitcast %[[INFBITS]] : i16 to bf16
    // CHECK: %[[E:.*]] = llvm.and %{{.*}}, %[[EXPMASK]] : i8
    // CHECK: %[[ISRSVD:.*]] = llvm.icmp "eq" %[[E]], %[[EXPMASK]] : i8
    // CHECK: %[[M:.*]] = llvm.and %{{.*}}, %[[MANTMASK]] : i8
    // CHECK: %[[ISNAN:.*]] = llvm.icmp "ne" %[[M]], %{{.*}} : i8
    // CHECK: %[[SPECIAL:.*]] = llvm.select %[[ISNAN]], %[[NAN]], %[[INF]] : i1, bf16
    // CHECK: %[[SPBITS:.*]] = llvm.bitcast %[[SPECIAL]] : bf16 to i16
    // CHECK: %[[SPSIGN:.*]] = llvm.or %[[SPBITS]], %{{.*}} : i16
    // CHECK: %[[SPBF:.*]] = llvm.bitcast %[[SPSIGN]] : i16 to bf16
    // CHECK: llvm.select %[[ISRSVD]], %[[SPBF]], %{{.*}} : i1, bf16
    // CHECK-NOT: llvm.extractelement {{.*}}vector<4xi32>
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}

// -----

// COM: LTS path — pure-integer table-lookup fallback (no ttig.support_bfloat16_arithmetic)
#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  // CHECK-LABEL: @convert_fp8e5m2_to_bf16_table
  tt.func public @convert_fp8e5m2_to_bf16_table(%src: tensor<16xf8E5M2, #blocked>) -> tensor<16xbf16, #blocked> {
    %dst = tt.fp_to_fp %src : tensor<16xf8E5M2, #blocked> -> tensor<16xbf16, #blocked>
    // CHECK-DAG: llvm.mlir.constant(260046848 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(939524096 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(931135488 : i32) : i32
    // CHECK-DAG: llvm.mlir.constant(943718400 : i32) : i32
    // CHECK: llvm.icmp "eq" {{.*}} : i32
    // CHECK: llvm.extractelement {{.*}}vector<4xi32>
    // CHECK: llvm.add {{.*}} : i32
    // CHECK: llvm.select {{.*}} : i1, i32
    // CHECK-NOT: llvm.fmul
    // CHECK-NOT: llvm.mlir.constant(5.192300e+33 : bf16)
    tt.return %dst : tensor<16xbf16, #blocked>
  }
}
