// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-intel-fold-true-cmpi -canonicalize | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @cmpsle(%arg0: !tt.ptr<f32>) -> i1 {
    %c0 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cmpsle = arith.cmpi sle, %c0, %c1024_i32 : i32
    tt.return %cmpsle: i1
  }
}

// CHECK-LABEL:   tt.func @cmpsle(
// CHECK-SAME:                       %[[VAL_0:.*]]: !tt.ptr<f32>) -> i1 {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           tt.return %[[VAL_1]] : i1
// CHECK:         }

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @assumepid(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c0 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %cmpsle = arith.cmpi sle, %pid, %c1024_i32 : i32
    llvm.intr.assume %cmpsle : i1
    %cmpsge = arith.cmpi sge, %pid, %c0 : i32
    llvm.intr.assume %cmpsge : i1
    %1 = arith.muli %pid, %c1024_i32 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// CHECK-LABEL:   tt.func @assumepid(
// CHECK-SAME:                       %[[VAL_0:.*]]: !tt.ptr<f32>) -> tensor<1024xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant true
// CHECK:           %[[VAL_2:.*]] = arith.constant 1024 : i32
// CHECK:           %[[VAL_3:.*]] = tt.get_program_id x : i32
// CHECK:           llvm.intr.assume %[[VAL_1]] : i1
// CHECK:           llvm.intr.assume %[[VAL_1]] : i1
// CHECK:           %[[VAL_4:.*]] = arith.muli %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_5:.*]] = tt.addptr %[[VAL_0]], %[[VAL_4]] : !tt.ptr<f32>, i32
// CHECK:           %[[VAL_6:.*]] = tt.splat %[[VAL_5]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_7:.*]] = tt.load %[[VAL_6]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_7]] : tensor<1024xf32>
// CHECK:         }

// -----

// COM: The third test case (assume_matmul) from the AMD version is dropped
// COM: because it uses NVIDIA MMA encodings (#ttg.nvidia_mma) which are
// COM: not relevant to Intel.

// -----

// Tensor-typed cmpi that is statically true should be folded to dense<true>.
// make_range(129, 257) elements are [129..256], make_range(0, 128) elements
// are [0..127]. Since min(t1)=129 > max(t0)=127, sgt is always true.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @foldtensorcmpi() -> tensor<128xi1> {
    %t0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %t1 = tt.make_range {end = 257 : i32, start = 129 : i32} : tensor<128xi32>
    %cmp = arith.cmpi sgt, %t1, %t0 : tensor<128xi32>
    tt.return %cmp: tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func @foldtensorcmpi
// CHECK:           %[[TRUE:.*]] = arith.constant dense<true> : tensor<128xi1>
// CHECK:           tt.return %[[TRUE]] : tensor<128xi1>
// CHECK:         }

// -----

// Tensor-typed cmpi that is statically false should be folded to dense<false>.
// make_range(0, 128) elements are [0..127], make_range(129, 257) elements
// are [129..256]. Since max(t0)=127 < min(t1)=129, sgt is always false.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @foldtensorcmpifalse() -> tensor<128xi1> {
    %t0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %t1 = tt.make_range {end = 257 : i32, start = 129 : i32} : tensor<128xi32>
    %cmp = arith.cmpi sgt, %t0, %t1 : tensor<128xi32>
    tt.return %cmp: tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func @foldtensorcmpifalse
// CHECK:           %[[FALSE:.*]] = arith.constant dense<false> : tensor<128xi1>
// CHECK:           tt.return %[[FALSE]] : tensor<128xi1>
// CHECK:         }

// -----

// Scalar cmpi that is statically false should be folded.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @cmpsle_false(%arg0: !tt.ptr<f32>) -> i1 {
    %c1024 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : i32
    %cmpsle = arith.cmpi sle, %c1024, %c0 : i32
    tt.return %cmpsle: i1
  }
}

// CHECK-LABEL:   tt.func @cmpsle_false(
// CHECK:           %[[FALSE:.*]] = arith.constant false
// CHECK:           tt.return %[[FALSE]] : i1
// CHECK:         }

// -----

// Tensor-typed cmpi that is NOT statically determinable should NOT be folded.
// make_range(0, 128) elements [0..127] vs make_range(64, 192) elements
// [64..191]. Ranges overlap, so slt is not always true or always false.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @dontfoldtensorcmpi() -> tensor<128xi1> {
    %t0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %t1 = tt.make_range {end = 192 : i32, start = 64 : i32} : tensor<128xi32>
    %cmp = arith.cmpi slt, %t0, %t1 : tensor<128xi32>
    tt.return %cmp: tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func @dontfoldtensorcmpi
// CHECK-NOT:       arith.constant dense<true>
// CHECK-NOT:       arith.constant dense<false>
// CHECK:           arith.cmpi slt
// CHECK:         }

// -----

// Tensor-typed cmpi with splat constant: [0..31] < 1024 is always true.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @foldtensorsplatcmpi() -> tensor<32xi1> {
    %t0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c1024 = arith.constant dense<1024> : tensor<32xi32>
    %cmp = arith.cmpi slt, %t0, %c1024 : tensor<32xi32>
    tt.return %cmp: tensor<32xi1>
  }
}

// CHECK-LABEL:   tt.func @foldtensorsplatcmpi
// CHECK:           %[[TRUE:.*]] = arith.constant dense<true> : tensor<32xi1>
// CHECK:           tt.return %[[TRUE]] : tensor<32xi1>
// CHECK:         }
