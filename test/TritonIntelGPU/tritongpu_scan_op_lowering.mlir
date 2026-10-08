// RUN: triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,SCAN-ON
// RUN: env TRITON_INTEL_SUBGROUP_SCAN=0 triton-opt %s --split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,SCAN-OFF

// COM: `tt.scan` lowers to the hardware sub-group scan when the scan axis covers
// COM: every lane of the sub-group, and to the generic shuffle chain otherwise.
// COM: The group operation operand is checked explicitly, not just the symbol:
// COM: Reduce/InclusiveScan/ExclusiveScan share one symbol and differ only in
// COM: that constant (0/1/2), so a symbol-only check would accept a wrong scan.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_1d_i32_add
  tt.func public @scan_1d_i32_add(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // SCAN-ON:          [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT:     [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT:     llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    // SCAN-ON-NOT:      sub_group_shuffle_up
    // SCAN-OFF-NOT:     GroupNonUniform
    // SCAN-OFF-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.addi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }
}

// -----

// COM: `reverse = true` flips the lane order before and after the in-warp phase.
// COM: A forward inclusive scan between the two flips is the reverse scan.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_1d_f32_add_reverse
  tt.func public @scan_1d_f32_add_reverse(%f : tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // SCAN-ON:      [[FLIP:%.*]] = llvm.call spir_funccc @_Z21sub_group_shuffle_xorfj
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: [[SCANNED:%.*]] = llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif([[SUBGROUP]], [[INCLUSIVE]], [[FLIP]]) {{.*}} : (i32, i32, f32) -> f32
    // SCAN-ON:      llvm.call spir_funccc @_Z21sub_group_shuffle_xorfj([[SCANNED]], %{{.*}})
    // SCAN-ON-NOT:  sub_group_shuffle_up
    %g = "tt.scan" (%f) ({
    ^bb0(%a: f32, %b: f32):
      %r = arith.addf %a, %b : f32
      tt.scan.return %r : f32
    }) {axis = 0 : i32, reverse = true} : (tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xf32, #blocked>
  }
}

// -----

// COM: More than one element per thread: the intra-thread scan runs first, then a
// COM: single builtin scans the per-thread totals. The one remaining shuffle_up is
// COM: the intra-thread carry, not part of the in-warp scan.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_1d_multi_elem_per_thread
  tt.func public @scan_1d_multi_elem_per_thread(%f : tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    // SCAN-ON:         [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT:    [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT:    llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    // SCAN-ON-COUNT-1: llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    // SCAN-ON-NOT:     sub_group_shuffle_up
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.addi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<128xi32, #blocked>
  }
}

// -----

// COM: More than one warp along the scan axis: the builtin replaces the in-warp
// COM: phase only, the cross-warp phase still goes through shared memory.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_1d_multi_axis_warp
  tt.func public @scan_1d_multi_axis_warp(%f : tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: [[WARPTOTAL:%.*]] = llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIAddiij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    // SCAN-ON:      llvm.store [[WARPTOTAL]], %{{.*}} : i32, !llvm.ptr<3>
    // SCAN-ON:      llvm.call spir_funccc @_Z7barrierj
    // SCAN-ON:      llvm.load %{{.*}} : !llvm.ptr<3> -> i32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.addi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<128xi32, #blocked>
  }
}

// -----

// COM: Op and element type coverage. `i1` uses the logical group ops; add,
// COM: maxsi and minsi are rejected instead, see the negative cases below.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_f32_mul
  tt.func public @scan_f32_mul(%f : tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFMuliif([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, f32) -> f32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: f32, %b: f32):
      %r = arith.mulf %a, %b : f32
      tt.scan.return %r : f32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xf32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i32_mul
  tt.func public @scan_i32_mul(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformIMuliij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.muli %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i32_maxsi
  tt.func public @scan_i32_maxsi(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformSMaxiij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.maxsi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i32_minui
  tt.func public @scan_i32_minui(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z27__spirv_GroupNonUniformUMiniij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.minui %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i32_andi
  tt.func public @scan_i32_andi(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z33__spirv_GroupNonUniformBitwiseAndiij([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i32) -> i32
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.andi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_andi
  tt.func public @scan_i1_andi(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z33__spirv_GroupNonUniformLogicalAndiib([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i1) -> i1
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.andi %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_ori
  tt.func public @scan_i1_ori(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z32__spirv_GroupNonUniformLogicalOriib([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i1) -> i1
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.ori %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_xori
  tt.func public @scan_i1_xori(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z33__spirv_GroupNonUniformLogicalXoriib([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i1) -> i1
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.xori %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_muli
  tt.func public @scan_i1_muli(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // SCAN-ON:      [[SUBGROUP:%.*]] = llvm.mlir.constant(3 : i32) : i32
    // SCAN-ON-NEXT: [[INCLUSIVE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // SCAN-ON-NEXT: llvm.call spir_funccc @_Z33__spirv_GroupNonUniformLogicalAndiib([[SUBGROUP]], [[INCLUSIVE]], %{{.*}}) {{.*}} : (i32, i32, i1) -> i1
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.muli %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }
}

// -----

// COM: Negative: the scan axis holds 16 of the 32 lanes. The builtin scans the
// COM: whole sub-group all-or-nothing and SPIR-V has no portable clustered scan,
// COM: so the shuffle chain must be kept.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_2d_subwarp_axis
  tt.func public @scan_2d_subwarp_axis(%f : tensor<2x16xi32, #blocked>) -> tensor<2x16xi32, #blocked> {
    // CHECK-NOT:      GroupNonUniform
    // CHECK-COUNT-4:  llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.addi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 1 : i32, reverse = false} : (tensor<2x16xi32, #blocked>) -> tensor<2x16xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<2x16xi32, #blocked>
  }
}

// -----

// COM: Negative: only 8 lanes hold unique data even though the encoding names 32
// COM: threads per warp, so the gate must use the unique-data lane count.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_1d_subwarp_axis
  tt.func public @scan_1d_subwarp_axis(%f : tensor<8xi32, #blocked>) -> tensor<8xi32, #blocked> {
    // CHECK-NOT:     GroupNonUniform
    // CHECK-COUNT-3: llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %r = arith.addi %a, %b : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<8xi32, #blocked>) -> tensor<8xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<8xi32, #blocked>
  }
}

// -----

// COM: Negative: a tuple scan would need one builtin per component; not supported.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_multi_operand
  tt.func public @scan_multi_operand(%f : tensor<32xi32, #blocked>, %h : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // CHECK-NOT:      GroupNonUniform
    // CHECK-COUNT-10: llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    %g:2 = "tt.scan" (%f, %h) ({
    ^bb0(%a0: i32, %a1: i32, %b0: i32, %b1: i32):
      %r0 = arith.addi %a0, %b0 : i32
      %r1 = arith.addi %a1, %b1 : i32
      tt.scan.return %r0, %r1 : i32, i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>, tensor<32xi32, #blocked>) -> (tensor<32xi32, #blocked>, tensor<32xi32, #blocked>)
    // CHECK: llvm.return
    tt.return %g#0 : tensor<32xi32, #blocked>
  }
}

// -----

// COM: Negative: the combine region is not a single binary op on the block args.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_select_combine
  tt.func public @scan_select_combine(%f : tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    // CHECK-NOT:     GroupNonUniform
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upij
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i32, %b: i32):
      %c = arith.cmpi slt, %a, %b : i32
      %r = arith.select %c, %b, %a : i32
      tt.scan.return %r : i32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi32, #blocked>
  }
}

// -----

// COM: Negative: the scan gate rejects these three `i1` combines.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_addi
  tt.func public @scan_i1_addi(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // CHECK-NOT:     GroupNonUniform
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upcj
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.addi %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_maxsi
  tt.func public @scan_i1_maxsi(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // CHECK-NOT:     GroupNonUniform
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upcj
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.maxsi %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_i1_minsi
  tt.func public @scan_i1_minsi(%f : tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked> {
    // CHECK-NOT:     GroupNonUniform
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upcj
    %g = "tt.scan" (%f) ({
    ^bb0(%a: i1, %b: i1):
      %r = arith.minsi %a, %b : i1
      tt.scan.return %r : i1
    }) {axis = 0 : i32, reverse = false} : (tensor<32xi1, #blocked>) -> tensor<32xi1, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xi1, #blocked>
  }
}

// -----

// COM: Negative: `arith.maxnumf`/`minnumf` return the numeric operand when exactly
// COM: one operand is NaN, but SPIR-V leaves the group FMax/FMin choice undefined
// COM: there. Rejected rather than lowered to an undefined-on-NaN builtin.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @scan_f32_maxnumf
  tt.func public @scan_f32_maxnumf(%f : tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // CHECK-NOT:     GroupNonUniformFMax
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upfj
    %g = "tt.scan" (%f) ({
    ^bb0(%a: f32, %b: f32):
      %r = arith.maxnumf %a, %b : f32
      tt.scan.return %r : f32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xf32, #blocked>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @scan_f32_minnumf
  tt.func public @scan_f32_minnumf(%f : tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // CHECK-NOT:     GroupNonUniformFMin
    // CHECK-COUNT-5: llvm.call spir_funccc @_Z20sub_group_shuffle_upfj
    %g = "tt.scan" (%f) ({
    ^bb0(%a: f32, %b: f32):
      %r = arith.minnumf %a, %b : f32
      tt.scan.return %r : f32
    }) {axis = 0 : i32, reverse = false} : (tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked>
    // CHECK: llvm.return
    tt.return %g : tensor<32xf32, #blocked>
  }
}
