// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer=native-gather-classes=false --split-input-file | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_GATHER=1 triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s --check-prefix=CLS
// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer=num-warps=8 --split-input-file | FileCheck %s --check-prefix=NW8
// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer=num-warps=2 --split-input-file | FileCheck %s --check-prefix=NW2

// COM: A gather on a device-side descriptor stays native (later lowered as
// COM: ttig.descriptor_gather) when the descriptor uses zero padding, the elements
// COM: are at least 8 bits wide (not i1), the last stride is 1, the column offset is
// COM: the constant 0 and the row access is proven 16-byte aligned. Otherwise its
// COM: makers are expanded and the gather becomes a masked tt.load, as before.
// COM: The first RUN line checks these rules with the tile-class rule off; CLS, NW2 and NW8
// COM: check the class rule (last section) at 4, 2 and 8 warps.

// COM: The vLLM paged-KV shape: the maker sits in an scf.if, the gather in an
// COM: scf.for inside it.
module {
  tt.func public @native_gather(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %cond: i1, %n: i32) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xbf16>
    %r = scf.if %cond -> (tensor<32x64xbf16>) {
      %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c64_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
      %acc = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%a = %zero) -> (tensor<32x64xbf16>) : i32 {
        %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
        %s = arith.addf %a, %g : tensor<32x64xbf16>
        scf.yield %s : tensor<32x64xbf16>
      }
      scf.yield %acc : tensor<32x64xbf16>
    } else {
      scf.yield %zero : tensor<32x64xbf16>
    }
    tt.return %r : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @native_gather
// CHECK:       tt.make_tensor_descriptor
// CHECK:       scf.for
// CHECK:       tt.descriptor_gather
// CHECK-NOT:   tt.load
// OFF-LABEL:   @native_gather
// OFF-NOT:     tt.descriptor_gather
// OFF:         tt.load

// -----

// COM: NaN padding: the native lowering fills out-of-bounds elements with zero.
module {
  tt.func public @nan_padding(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] {padding = 2 : i32} : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @nan_padding
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load

// -----

// COM: Divergent padding: the descriptor comes from a zero-padded maker on one
// COM: path and a NaN-padded one on the other.
module {
  tt.func public @divergent_padding(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %cond: i1) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = scf.if %cond -> (!tt.tensordesc<1x64xbf16>) {
      %d0 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] {padding = 1 : i32} : <bf16>, <1x64xbf16>
      scf.yield %d0 : !tt.tensordesc<1x64xbf16>
    } else {
      %d1 = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] {padding = 2 : i32} : <bf16>, <1x64xbf16>
      scf.yield %d1 : !tt.tensordesc<1x64xbf16>
    }
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @divergent_padding
// CHECK-NOT:   tt.make_tensor_descriptor
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load

// -----

// COM: i1 elements: the native lowering has no bool-as-byte handling.
module {
  tt.func public @i1_gather(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x256xi1> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <i1>, <1x256xi1>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x256xi1>, tensor<32xi32>, i32) -> tensor<32x256xi1>
    tt.return %g : tensor<32x256xi1>
  }
}

// CHECK-LABEL: @i1_gather
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load

// -----

// COM: Non-unit last stride: the vectorized lowering loads adjacent columns.
module {
  tt.func public @strided_columns(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c1_i64, %c256_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @strided_columns
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load

// -----

// COM: A maker shared by a descriptor load and a gather keeps both native.
module {
  tt.func public @shared_with_load(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> (tensor<32x64xbf16>, tensor<1x64xbf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    %l = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<1x64xbf16> -> tensor<1x64xbf16>
    tt.return %g, %l : tensor<32x64xbf16>, tensor<1x64xbf16>
  }
}

// CHECK-LABEL: @shared_with_load
// CHECK:       tt.descriptor_gather
// CHECK:       tt.descriptor_load
// CHECK-NOT:   tt.load

// -----

// COM: A maker shared with a scatter (not dot-fed, so not kept native) expands for both.
module {
  tt.func public @shared_with_scatter(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.descriptor_scatter %desc[%idx, %c0_i32], %g : !tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32, tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @shared_with_scatter
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load
// CHECK:       tt.store

// -----

// COM: A loop-carried descriptor that traces to one eligible maker stays native.
module {
  tt.func public @loop_carried(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %n: i32) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xbf16>
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %r:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%d = %desc, %a = %zero) -> (!tt.tensordesc<1x64xbf16>, tensor<32x64xbf16>) : i32 {
      %g = tt.descriptor_gather %d[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
      %s = arith.addf %a, %g : tensor<32x64xbf16>
      scf.yield %d, %s : !tt.tensordesc<1x64xbf16>, tensor<32x64xbf16>
    }
    tt.return %r#1 : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @loop_carried
// CHECK:       scf.for {{.*}} -> (!tt.tensordesc<1x64xbf16>, tensor<32x64xbf16>)
// CHECK:       tt.descriptor_gather
// CHECK-NOT:   tt.load

// -----

// COM: One function per admission rule; each is admissible but for its rule.
module {
  // COM: A runtime column offset is rejected, even one proven 16-aligned.
  tt.func public @runtime_y(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %y: i32 {tt.divisibility = 16 : i32}) -> tensor<32x64xbf16> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %y] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
  // COM: A nonzero constant column offset is rejected.
  tt.func public @const_y16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
    %c16_i32 = arith.constant 16 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c16_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
  // COM: A shape hint of 12 has power-of-two divisor 4 (8 bytes) and is rejected.
  tt.func public @shape_div12(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %w: i32 {tt.divisibility = 12 : i32}) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %w], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
  // COM: A shape hint of 24 has divisor 8 (16 bytes) and is admitted.
  tt.func public @shape_hint24(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %w: i32 {tt.divisibility = 24 : i32}) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %w], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    tt.return %g : tensor<32x64xbf16>
  }
  // COM: Alignment is read at the gather: a loop that starts on an 8-byte base
  // COM: rejects it, although the descriptor it carries afterwards is aligned.
  tt.func public @joined_loop(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 8 : i32}, %idx: tensor<32xi32>, %n: i32) -> tensor<32x64xbf16> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c256_i32 = arith.constant 256 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xbf16>
    %desc = tt.make_tensor_descriptor %arg1, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %r:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%d = %desc, %a = %zero) -> (!tt.tensordesc<1x64xbf16>, tensor<32x64xbf16>) : i32 {
      %g = tt.descriptor_gather %d[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
      %s = arith.addf %a, %g : tensor<32x64xbf16>
      %nd = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c64_i64, %c1_i64] : <bf16>, <1x64xbf16>
      scf.yield %nd, %s : !tt.tensordesc<1x64xbf16>, tensor<32x64xbf16>
    }
    tt.return %r#1 : tensor<32x64xbf16>
  }
}

// CHECK-LABEL: @runtime_y
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load
// CHECK-LABEL: @const_y16
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load
// CHECK-LABEL: @shape_div12
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load
// CHECK-LABEL: @shape_hint24
// CHECK:       tt.descriptor_gather
// CHECK-NOT:   tt.load
// CHECK-LABEL: @joined_loop
// CHECK-NOT:   tt.descriptor_gather
// CHECK:       tt.load

// -----

// COM: One function per class rule; bytes are per warp (num-warps 4 by default, 2 and 8 under NW2 and NW8).
module {
  // COM: An f16/bf16 tile used only as a tt.dot A operand is native on 2 or 4 warps ...
  tt.func public @dot16_1k(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %b: tensor<128x64xbf16>) -> tensor<32x64xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<32xi32>, i32) -> tensor<32x128xbf16>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x64xf32>
    %d = tt.dot %g, %b, %acc : tensor<32x128xbf16> * tensor<128x64xbf16> -> tensor<32x64xf32>
    tt.return %d : tensor<32x64xf32>
  }

  // COM: ... up to 8 KiB per warp ...
  tt.func public @dot16_8k(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<128xi32>, %b: tensor<128x16xbf16>) -> tensor<128x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<128xi32>, i32) -> tensor<128x128xbf16>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x16xf32>
    %d = tt.dot %g, %b, %acc : tensor<128x128xbf16> * tensor<128x16xbf16> -> tensor<128x16xf32>
    tt.return %d : tensor<128x16xf32>
  }

  // COM: A B operand is not native ...
  tt.func public @dotB16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<64xi32>, %p: tensor<16x64xbf16>) -> tensor<16x64xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 64 : i64
    %cw_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<64xi32>, i32) -> tensor<64x64xbf16>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %d = tt.dot %p, %g, %acc : tensor<16x64xbf16> * tensor<64x64xbf16> -> tensor<16x64xf32>
    tt.return %d : tensor<16x64xf32>
  }

  // COM: ... nor an A operand that is also stored.
  tt.func public @dotA_and_store16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %b: tensor<64x16xbf16>, %out: !tt.ptr<bf16>) -> tensor<32x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 64 : i64
    %cw_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <bf16>, <1x64xbf16>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xbf16>, tensor<32xi32>, i32) -> tensor<32x64xbf16>
    %p = tt.splat %out : !tt.ptr<bf16> -> tensor<32x64x!tt.ptr<bf16>>
    tt.store %p, %g : tensor<32x64x!tt.ptr<bf16>>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
    %d = tt.dot %g, %b, %acc : tensor<32x64xbf16> * tensor<64x16xbf16> -> tensor<32x16xf32>
    tt.return %d : tensor<32x16xf32>
  }

  // COM: Only f16/bf16 dot operands were measured.
  tt.func public @f32_dot(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>, %b: tensor<64x16xf32>) -> tensor<32x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 64 : i64
    %cw_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f32>, <1x64xf32>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x64xf32>, tensor<32xi32>, i32) -> tensor<32x64xf32>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x16xf32>
    %d = tt.dot %g, %b, %acc : tensor<32x64xf32> * tensor<64x16xf32> -> tensor<32x16xf32>
    tt.return %d : tensor<32x16xf32>
  }

  // COM: An f8E4M3FN tile converted by tt.fp_to_fp is native on 4 warps at 2-4 KiB per warp: not at 1 KiB ...
  tt.func public @fp8_cvt_1k(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<32xi32>) -> tensor<32x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x128xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E4M3FN>, tensor<32xi32>, i32) -> tensor<32x128xf8E4M3FN>
    %f = tt.fp_to_fp %g : tensor<32x128xf8E4M3FN> -> tensor<32x128xf32>
    tt.return %f : tensor<32x128xf32>
  }

  // COM: ... at 2 KiB ...
  tt.func public @fp8_cvt_2k(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<64xi32>) -> tensor<64x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x128xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E4M3FN>, tensor<64xi32>, i32) -> tensor<64x128xf8E4M3FN>
    %f = tt.fp_to_fp %g : tensor<64x128xf8E4M3FN> -> tensor<64x128xf32>
    tt.return %f : tensor<64x128xf32>
  }

  // COM: ... and 4 KiB ...
  tt.func public @fp8_cvt_4k(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<128xi32>) -> tensor<128x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x128xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E4M3FN>, tensor<128xi32>, i32) -> tensor<128x128xf8E4M3FN>
    %f = tt.fp_to_fp %g : tensor<128x128xf8E4M3FN> -> tensor<128x128xf32>
    tt.return %f : tensor<128x128xf32>
  }

  // COM: ... and not at 8 KiB.
  tt.func public @fp8_cvt_8k(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<128xi32>) -> tensor<128x256xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 256 : i64
    %cw_i32 = arith.constant 256 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x256xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x256xf8E4M3FN>, tensor<128xi32>, i32) -> tensor<128x256xf8E4M3FN>
    %f = tt.fp_to_fp %g : tensor<128x256xf8E4M3FN> -> tensor<128x256xf32>
    tt.return %f : tensor<128x256xf32>
  }

  // COM: An f8E4M3FN tile used only by a dot is not native.
  tt.func public @fp8_dot(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<64xi32>, %b: tensor<128x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x128xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E4M3FN>, tensor<64xi32>, i32) -> tensor<64x128xf8E4M3FN>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x16xf32>
    %d = tt.dot %g, %b, %acc : tensor<64x128xf8E4M3FN> * tensor<128x16xf8E4M3FN> -> tensor<64x16xf32>
    tt.return %d : tensor<64x16xf32>
  }

  // COM: ... nor one that is also stored.
  tt.func public @fp8_cvt_and_store(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %idx: tensor<64xi32>, %out: !tt.ptr<f8E4M3FN>) -> tensor<64x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E4M3FN>, <1x128xf8E4M3FN>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E4M3FN>, tensor<64xi32>, i32) -> tensor<64x128xf8E4M3FN>
    %p = tt.splat %out : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
    tt.store %p, %g : tensor<64x128x!tt.ptr<f8E4M3FN>>
    %f = tt.fp_to_fp %g : tensor<64x128xf8E4M3FN> -> tensor<64x128xf32>
    tt.return %f : tensor<64x128xf32>
  }

  // COM: Only f8E4M3FN was measured.
  tt.func public @fp8e5_cvt(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %idx: tensor<64xi32>) -> tensor<64x128xf32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %cw_i64 = arith.constant 128 : i64
    %cw_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %cw_i32], [%cw_i64, %c1_i64] : <f8E5M2>, <1x128xf8E5M2>
    %g = tt.descriptor_gather %desc[%idx, %c0_i32] : (!tt.tensordesc<1x128xf8E5M2>, tensor<64xi32>, i32) -> tensor<64x128xf8E5M2>
    %f = tt.fp_to_fp %g : tensor<64x128xf8E5M2> -> tensor<64x128xf32>
    tt.return %f : tensor<64x128xf32>
  }
}

// CLS-LABEL: @dot16_1k
// CLS:       tt.descriptor_gather
// CLS-LABEL: @dot16_8k
// CLS:       tt.descriptor_gather
// CLS-LABEL: @dotB16
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @dotA_and_store16
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @f32_dot
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @fp8_cvt_1k
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @fp8_cvt_2k
// CLS:       tt.descriptor_gather
// CLS-LABEL: @fp8_cvt_4k
// CLS:       tt.descriptor_gather
// CLS-LABEL: @fp8_cvt_8k
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @fp8_dot
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @fp8_cvt_and_store
// CLS-NOT:   tt.descriptor_gather
// CLS-LABEL: @fp8e5_cvt
// CLS-NOT:   tt.descriptor_gather
// NW2-LABEL: @dot16_1k
// NW2:       tt.descriptor_gather
// NW2-LABEL: @dot16_8k
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @dotB16
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @dotA_and_store16
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @f32_dot
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8_cvt_1k
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8_cvt_2k
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8_cvt_4k
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8_cvt_8k
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8_dot
// NW2-NOT:   tt.descriptor_gather
// NW2-LABEL: @fp8e5_cvt
// NW2-NOT:   tt.descriptor_gather
// NW8-LABEL: @dot16_1k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @dot16_8k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @dotB16
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @dotA_and_store16
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @f32_dot
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8_cvt_1k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8_cvt_2k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8_cvt_4k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8_cvt_8k
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8_dot
// NW8-NOT:   tt.descriptor_gather
// NW8-LABEL: @fp8e5_cvt
// NW8-NOT:   tt.descriptor_gather
