// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s

// COM: End to end from `tt.load` / `tt.descriptor_load`, so that each default
// COM: value reaches the `triton_gen.predicated_load` lowering through the chain
// COM: the load lowering really builds, rather than a hand-written one.
// COM:
// COM: IGC selects a predicated load's result into the register holding its
// COM: default value, so predicated loads sharing one non-constant default
// COM: clobber each other (#8121). A non-constant default is therefore never
// COM: passed to the builtin: the load takes a zero default and the real one is
// COM: applied with a select. A constant default is passed straight through.
// COM: Every check is anchored on the value's consumer, the bitcast back to f32.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttig.support_predicated_io} {
  // COM: Non-constant `other`: each of this work item's four loads takes a zero
  // COM: default, and `other` is applied with a select on the same predicate.
  // CHECK-LABEL: masked_load_runtime_other
  tt.func @masked_load_runtime_other(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>, %other: tensor<256xf32, #blocked>, %dst: tensor<256x!tt.ptr<f32>, #blocked>) {
    // CHECK:      [[W0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[Z0:%.*]] = llvm.mlir.zero : i32
    // CHECK-NEXT: [[L0:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, [[P0:%[0-9]+]], [[Z0]])
    // CHECK-NEXT: [[S0:%.*]] = llvm.select [[P0]], [[L0]], [[W0]] : i1, i32
    // CHECK-NEXT: llvm.bitcast [[S0]] : i32 to f32
    // CHECK:      [[W1:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[Z1:%.*]] = llvm.mlir.zero : i32
    // CHECK-NEXT: [[L1:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, [[P1:%[0-9]+]], [[Z1]])
    // CHECK-NEXT: [[S1:%.*]] = llvm.select [[P1]], [[L1]], [[W1]] : i1, i32
    // CHECK-NEXT: llvm.bitcast [[S1]] : i32 to f32
    // CHECK:      [[W2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[Z2:%.*]] = llvm.mlir.zero : i32
    // CHECK-NEXT: [[L2:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, [[P2:%[0-9]+]], [[Z2]])
    // CHECK-NEXT: [[S2:%.*]] = llvm.select [[P2]], [[L2]], [[W2]] : i1, i32
    // CHECK-NEXT: llvm.bitcast [[S2]] : i32 to f32
    // CHECK:      [[W3:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[Z3:%.*]] = llvm.mlir.zero : i32
    // CHECK-NEXT: [[L3:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, [[P3:%[0-9]+]], [[Z3]])
    // CHECK-NEXT: [[S3:%.*]] = llvm.select [[P3]], [[L3]], [[W3]] : i1, i32
    // CHECK-NEXT: llvm.bitcast [[S3]] : i32 to f32
    %val = tt.load %ptr, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %dst, %val : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttig.support_predicated_io} {
  // COM: Constant `other` tensor: no select. Its value reaches the lowering
  // COM: through the tensor's unpacked struct (insertvalue/extractvalue) and then
  // COM: insertelement/bitcast, so this is what fails if any of those hops is
  // COM: not looked through.
  // CHECK-LABEL: masked_load_constant_other
  tt.func @masked_load_constant_other(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>, %dst: tensor<256x!tt.ptr<f32>, #blocked>) {
    // CHECK-NOT: llvm.select
    // CHECK:      [[W0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L0:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W0]])
    // CHECK-NEXT: llvm.bitcast [[L0]] : i32 to f32
    // CHECK:      [[W1:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L1:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W1]])
    // CHECK-NEXT: llvm.bitcast [[L1]] : i32 to f32
    // CHECK:      [[W2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L2:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W2]])
    // CHECK-NEXT: llvm.bitcast [[L2]] : i32 to f32
    // CHECK:      [[W3:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L3:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W3]])
    // CHECK-NEXT: llvm.bitcast [[L3]] : i32 to f32
    // CHECK-NOT: llvm.select
    %other = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %val = tt.load %ptr, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %dst, %val : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttig.support_predicated_io} {
  // COM: No `other` at all: the default is a constant zero, so no select.
  // CHECK-LABEL: masked_load_no_other
  tt.func @masked_load_no_other(%ptr: tensor<64x!tt.ptr<f32>, #blocked>, %mask: tensor<64xi1, #blocked>, %dst: tensor<64x!tt.ptr<f32>, #blocked>) {
    // CHECK-NOT: llvm.select
    // CHECK:      [[L0:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{.*}})
    // CHECK-NEXT: llvm.bitcast [[L0]] : i32 to f32
    // CHECK-NOT: llvm.select
    %val = tt.load %ptr, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %dst, %val : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttig.support_predicated_io} {
  // COM: A descriptor load's boundary padding is constant, so all eight loads
  // COM: keep the direct predicated form (the path #7090 measured as faster).
  // CHECK-LABEL: descriptor_load_padding
  tt.func @descriptor_load_padding(%desc: !tt.tensordesc<256xf32>, %dst: tensor<256x!tt.ptr<f32>, #blocked0>) {
    // CHECK-NOT: llvm.select
    // CHECK:      [[W0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L0:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W0]])
    // CHECK-NEXT: llvm.bitcast [[L0]] : i32 to f32
    // CHECK:      [[W1:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L1:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W1]])
    // CHECK-NEXT: llvm.bitcast [[L1]] : i32 to f32
    // CHECK:      [[W2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L2:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W2]])
    // CHECK-NEXT: llvm.bitcast [[L2]] : i32 to f32
    // CHECK:      [[W3:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L3:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W3]])
    // CHECK-NEXT: llvm.bitcast [[L3]] : i32 to f32
    // CHECK:      [[W4:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L4:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W4]])
    // CHECK-NEXT: llvm.bitcast [[L4]] : i32 to f32
    // CHECK:      [[W5:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L5:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W5]])
    // CHECK-NEXT: llvm.bitcast [[L5]] : i32 to f32
    // CHECK:      [[W6:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L6:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W6]])
    // CHECK-NEXT: llvm.bitcast [[L6]] : i32 to f32
    // CHECK:      [[W7:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      [[L7:%.*]] = llvm.call spir_funccc @_Z27__spirv_PredicatedLoadINTELPU3AS1vbi({{%.*}}, {{%.*}}, [[W7]])
    // CHECK-NEXT: llvm.bitcast [[L7]] : i32 to f32
    // CHECK-NOT: llvm.select
    %c0_i32 = arith.constant 0 : i32
    %val = tt.descriptor_load %desc[%c0_i32] : !tt.tensordesc<256xf32> -> tensor<256xf32, #blocked0>
    tt.store %dst, %val : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}
