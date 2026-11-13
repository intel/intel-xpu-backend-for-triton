// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not='llvm.bitcast {{.*}}xi1>' --implicit-check-not='llvm.bitcast {{.*}} : i{{(8|16|32|64)}} to i1{{$}}' --implicit-check-not='llvm.sext {{.*}} : i1 to'

// Bool tensors are bytes in memory, as for tl.store: zext, then pack i8 words (#8670).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_i1
  // CHECK:       llvm.zext {{%.*}} : i1 to i8
  // CHECK:       [[VEC:%.*]] = llvm.insertelement {{%.*}}, {{%.*}}[{{%.*}} : i32] : vector<1xi8>
  // CHECK:       [[WORD:%.*]] = llvm.bitcast [[VEC]] : vector<1xi8> to i8
  // CHECK:       llvm.store [[WORD]], {{.*}} : i8, !llvm.ptr<1>
  tt.func public @store_i1(%ptr: !tt.ptr<i1>, %m: i32, %n: i32, %stride: i64, %v: tensor<1x16xi1, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %ptr, [%m, %n], [%stride, %c1_i64] : <i1>, <1x16xi1>
    tt.descriptor_store %desc[%c0, %c0], %v : !tt.tensordesc<1x16xi1>, tensor<1x16xi1, #blocked>
    tt.return
  }
}

// -----

// Loads read i8 words, fill with i8 0, and narrow each byte with != 0, as for tl.load.
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @load_i1
  // CHECK:       llvm.bitcast {{%.*}} : vector<4xi8> to i32
  // CHECK:       llvm.load {{.*}} -> i32
  // CHECK:       [[WORD:%.*]] = llvm.bitcast {{%.*}} : i32 to vector<4xi8>
  // CHECK:       llvm.extractelement [[WORD]][{{%.*}} : i32] : vector<4xi8>
  // CHECK:       llvm.icmp "ne" {{%.*}}, {{%.*}} : i8
  tt.func public @load_i1(%ptr: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %m: i32, %n: i32 {tt.divisibility = 16 : i32}, %stride: i64 {tt.divisibility = 16 : i32}) -> tensor<1x64xi1, #blocked4> {
    %c0 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %ptr, [%m, %n], [%stride, %c1_i64] : <i1>, <1x64xi1>
    %v = tt.descriptor_load %desc[%c0, %c0] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x64xi1> -> tensor<1x64xi1, #blocked4>
    tt.return %v : tensor<1x64xi1, #blocked4>
  }
}
