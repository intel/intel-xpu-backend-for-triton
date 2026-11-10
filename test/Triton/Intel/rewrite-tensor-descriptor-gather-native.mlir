// RUN: triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s
// RUN: env TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_GATHER=1 triton-opt %s --triton-intel-rewrite-tensor-descriptor-to-pointer --split-input-file | FileCheck %s --check-prefix=OFF

// COM: A gather on a device-side descriptor stays native (later lowered as
// COM: ttig.descriptor_gather) when the descriptor uses zero padding, the
// COM: elements are not i1 and the last stride is 1. Otherwise its makers are
// COM: expanded and the gather becomes a masked tt.load, as before.

// COM: The vLLM paged-KV shape: the maker sits in an scf.if, the gather in an
// COM: scf.for inside it.
module {
  tt.func public @native_gather(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>, %cond: i1, %n: i32) -> tensor<32x64xbf16> {
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
  tt.func public @nan_padding(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
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
  tt.func public @divergent_padding(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>, %cond: i1) -> tensor<32x64xbf16> {
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
  tt.func public @i1_gather(%arg0: !tt.ptr<i1>, %idx: tensor<32xi32>) -> tensor<32x256xi1> {
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

// COM: Non-unit last stride: the native fast paths load contiguous columns.
module {
  tt.func public @strided_columns(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
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
  tt.func public @shared_with_load(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>) -> (tensor<32x64xbf16>, tensor<1x64xbf16>) {
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

// COM: A maker shared with a scatter (no native lowering) expands for both.
module {
  tt.func public @shared_with_scatter(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>) -> tensor<32x64xbf16> {
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
  tt.func public @loop_carried(%arg0: !tt.ptr<bf16>, %idx: tensor<32xi32>, %n: i32) -> tensor<32x64xbf16> {
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
