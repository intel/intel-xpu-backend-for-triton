// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_LOAD_TO_LOAD=0 triton-opt %s -split-input-file -verify-diagnostics --tritonintelgpu-lower-to-2d-block-load

// COM: Descriptor loads the tt.load lowering refuses. The module has no 2D block IO, so every load takes it.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @no_encoding(%d: !tt.tensordesc<1x32xf16>, %i: i32) {
    // expected-error @+1 {{'tt.descriptor_load' op requires a distributed result encoding}}
    %x = tt.descriptor_load %d[%i, %i] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x32xf16> -> tensor<1x32xf16>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @column_major_rank1(%d: !tt.tensordesc<1x32xf16>, %i: i32) {
    // expected-error @+1 {{'tt.descriptor_load' op column_major load requires a result of rank >= 2}}
    %x = tt.descriptor_load %d[%i, %i] {ttig.block_io = "column_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x32xf16> -> tensor<32xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @descriptor_rank_below_result_rank(%d: !tt.tensordesc<32xf16>, %i: i32) {
    // expected-error @+1 {{'tt.descriptor_load' op descriptor rank must be >= result rank}}
    %x = tt.descriptor_load %d[%i] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<32xf16> -> tensor<1x32xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @dropped_dim_not_one(%d: !tt.tensordesc<2x16xf16>, %i: i32) {
    // expected-error @+1 {{rank-reducing descriptor op only supports dropping leading dimensions of size 1, but descriptor dim[0] = 2}}
    %x = tt.descriptor_load %d[%i, %i] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<2x16xf16> -> tensor<32xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @inner_shape_mismatch(%d: !tt.tensordesc<1x32xf16>, %i: i32) {
    // expected-error @+1 {{descriptor inner dim[0] = 1 != tensor dim[0] = 2}}
    %x = tt.descriptor_load %d[%i, %i] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x32xf16> -> tensor<2x16xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @address_spaces_disagree(%p: !tt.ptr<f16>, %q: !tt.ptr<f16, "constant">, %m: i32, %n: i32, %s: i64, %i: i32, %c: i1) {
    %c1_i64 = arith.constant 1 : i64
    %d0 = tt.make_tensor_descriptor %p, [%m, %n], [%s, %c1_i64] : <f16>, <1x32xf16>
    %d1 = tt.make_tensor_descriptor %q, [%m, %n], [%s, %c1_i64] : <f16, "constant">, <1x32xf16>
    %d = arith.select %c, %d0, %d1 : !tt.tensordesc<1x32xf16>
    // expected-error @+1 {{'tt.descriptor_load' op the operations defining this descriptor disagree on the address space of its base}}
    %x = tt.descriptor_load %d[%i, %i] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x32xf16> -> tensor<1x32xf16, #blocked>
    tt.return
  }
}
