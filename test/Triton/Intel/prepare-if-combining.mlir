// RUN: triton-opt %s -split-input-file -triton-intel-prepare-if-combining | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-intel-prepare-if-combining -canonicalize | FileCheck %s --check-prefix=CANON
// RUN: triton-opt %s -split-input-file -triton-intel-prepare-if-combining=max-cloned-ops=1 | FileCheck %s --check-prefix=CAP
// RUN: triton-opt %s -split-input-file -triton-intel-prepare-if-combining=max-cloned-ops=7 -canonicalize | FileCheck %s --check-prefix=BELOW
// RUN: triton-opt %s -split-input-file -triton-intel-prepare-if-combining=max-cloned-ops=8 -canonicalize | FileCheck %s --check-prefix=AT


// CHECK-LABEL: op_between_ifs
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: op_between_ifs
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @op_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %0 = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %1 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %2 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %3 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %2, %2 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %2, %2 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %4 = ttg.convert_layout %3 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %5 = arith.addf %4, %1 : tensor<32x32xf32, #blocked>
    tt.return %5 : tensor<32x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: multiple_ops_between_ifs
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: %[[CVT:.+]] = ttg.convert_layout %[[LOAD]]
//  CHECK-NEXT: tt.trans %[[CVT]]
//  CHECK-NEXT: scf.if
// The cap limits only copying into branches: two ops are still hoisted.
// CAP-LABEL: multiple_ops_between_ifs
//       CAP: %[[LOAD:.+]] = ttg.local_load
//  CAP-NEXT: %[[CVT:.+]] = ttg.convert_layout %[[LOAD]]
//  CAP-NEXT: tt.trans %[[CVT]]
//  CAP-NEXT: scf.if
// CANON-LABEL: multiple_ops_between_ifs
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @multiple_ops_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %0 = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %1 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %2 = ttg.convert_layout %0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %add = arith.addf %3, %3 : tensor<32x32xf32, #blocked>
      scf.yield %add : tensor<32x32xf32, #blocked>
    } else {
      %sub = arith.subf %3, %3 : tensor<32x32xf32, #blocked>
      scf.yield %sub : tensor<32x32xf32, #blocked>
    }
    %5 = arith.addf %4, %1 : tensor<32x32xf32, #blocked>
    tt.return %5 : tensor<32x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: op_between_ifs_inside_for
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//       CHECK: scf.for
//  CHECK-NEXT: tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: op_between_ifs_inside_for
//       CANON: scf.for
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @op_between_ifs_inside_for(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a) -> tensor<32x32xf32, #blocked> {
      %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
        %mul = arith.mulf %acc, %acc : tensor<32x32xf32, #blocked>
        scf.yield %mul : tensor<32x32xf32, #blocked>
      } else {
        %div = arith.divf %acc, %acc : tensor<32x32xf32, #blocked>
        scf.yield %div : tensor<32x32xf32, #blocked>
      }
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
      scf.yield %4 : tensor<32x32xf32, #blocked>
    }
    tt.return %result : tensor<32x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: sibling_ifs_with_nested_if
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: sibling_ifs_with_nested_if
//       CANON: scf.if %arg0
//       CANON:   scf.if %arg1
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sibling_ifs_with_nested_if(%cond: i1, %cond2: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %inner = scf.if %cond2 -> tensor<32x32xf32, #blocked> {
        %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
        scf.yield %mul : tensor<32x32xf32, #blocked>
      } else {
        scf.yield %a : tensor<32x32xf32, #blocked>
      }
      scf.yield %inner : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
    tt.return %4 : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: ifs in different blocks
// CHECK-LABEL: ifs_in_different_blocks
//       CHECK: scf.if
//       CHECK: scf.for
//  CHECK-NEXT: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: ifs_in_different_blocks
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ifs_in_different_blocks(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %0) -> tensor<32x32xf32, #blocked> {
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      %4 = arith.addf %3, %acc : tensor<32x32xf32, #blocked>
      scf.yield %4 : tensor<32x32xf32, #blocked>
    }
    tt.return %result : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: nested ifs
// CHECK-LABEL: nested_ifs
//       CHECK: scf.if
//  CHECK-NEXT: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: nested_ifs
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @nested_ifs(%cond: i1, %cond2: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond2 -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      scf.yield %3 : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: ifs with different conditions
// CHECK-LABEL: different_conditions
//       CHECK: scf.if
//  CHECK-NEXT: arith.mulf
//       CHECK: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: different_conditions
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @different_conditions(%cond1: i1, %cond2: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond1 -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %2 = scf.if %cond2 -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
    tt.return %4 : tensor<32x32xf32, #blocked>
  }
}

// -----

// A non-pure op (local_store) between the ifs: the pure tt.trans is hoisted,
// the local_store is copied into both branches of the first if.
// CHECK-LABEL: non_pure_op_between_ifs
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: %[[TRANS:.+]] = tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
//  CHECK-NEXT:   arith.mulf
//  CHECK-NEXT:   ttg.local_store %[[TRANS]]
//  CHECK-NEXT:   scf.yield
//  CHECK-NEXT: } else {
//  CHECK-NEXT:   arith.divf
//  CHECK-NEXT:   ttg.local_store %[[TRANS]]
//  CHECK-NEXT:   scf.yield
//  CHECK-NEXT: }
//  CHECK-NEXT: scf.if
// CANON-LABEL: non_pure_op_between_ifs
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_transposed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @non_pure_op_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %smem_trans: !ttg.memdesc<32x32xf32, #shared_transposed, #smem, mutable>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    ttg.local_store %1, %smem_trans : tensor<32x32xf32, #blocked_transposed> -> !ttg.memdesc<32x32xf32, #shared_transposed, #smem, mutable>
    %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
    tt.return %4 : tensor<32x32xf32, #blocked>
  }
}

// -----

// Three ifs on structurally identical guards, each guard materialized right
// before its own if (the shape RemoveMasks produces for layer norm), with a
// reduction of the previous if's result and stores between the ifs.
// CHECK-LABEL: @layer_norm_chain
//       CHECK: %[[G:.+]] = arith.cmpi eq
//       CHECK: %[[NF:.+]] = arith.sitofp %arg2
//  CHECK-NEXT: %[[XS:.+]] = tt.splat %arg0
//  CHECK-NEXT: %[[IF0:.+]]:3 = scf.if %[[G]] -> (tensor<128xf32>, f32, tensor<128xf32>) {
//  CHECK-NEXT:   %[[T0:.+]] = arith.addf %arg3, %arg4
//  CHECK-NEXT:   %[[S0:.+]] = "tt.reduce"(%[[T0]])
//       CHECK:   %[[M0:.+]] = arith.divf %[[S0]], %[[NF]]
//  CHECK-NEXT:   %[[V0:.+]] = tt.splat %[[M0]]
//  CHECK-NEXT:   scf.yield %[[T0]], %[[M0]], %[[V0]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT:   %[[E0:.+]] = arith.subf %arg3, %arg4
//  CHECK-NEXT:   %[[S1:.+]] = "tt.reduce"(%[[E0]])
//       CHECK:   %[[M1:.+]] = arith.divf %[[S1]], %[[NF]]
//  CHECK-NEXT:   %[[V1:.+]] = tt.splat %[[M1]]
//  CHECK-NEXT:   scf.yield %[[E0]], %[[M1]], %[[V1]]
//  CHECK-NEXT: }
//  CHECK-NEXT: %[[IF1:.+]]:2 = scf.if %[[G]] -> (tensor<128xf32>, tensor<128xf32>) {
//  CHECK-NEXT:   %[[T1:.+]] = arith.mulf %[[IF0]]#2, %arg3
//  CHECK-NEXT:   %[[S2:.+]] = "tt.reduce"(%[[T1]])
//       CHECK:   %[[VAR0:.+]] = arith.divf %[[S2]], %[[NF]]
//  CHECK-NEXT:   tt.store %arg1, %[[IF0]]#1
//  CHECK-NEXT:   tt.store %arg0, %[[VAR0]]
//  CHECK-NEXT:   %[[VV0:.+]] = tt.splat %[[VAR0]]
//  CHECK-NEXT:   scf.yield %[[T1]], %[[VV0]]
//  CHECK-NEXT: } else {
//  CHECK-NEXT:   %[[E1:.+]] = arith.mulf %[[IF0]]#2, %arg4
//  CHECK-NEXT:   %[[S3:.+]] = "tt.reduce"(%[[E1]])
//       CHECK:   %[[VAR1:.+]] = arith.divf %[[S3]], %[[NF]]
//  CHECK-NEXT:   tt.store %arg1, %[[IF0]]#1
//  CHECK-NEXT:   tt.store %arg0, %[[VAR1]]
//  CHECK-NEXT:   %[[VV1:.+]] = tt.splat %[[VAR1]]
//  CHECK-NEXT:   scf.yield %[[E1]], %[[VV1]]
//  CHECK-NEXT: }
//  CHECK-NEXT: scf.if %[[G]] {
//  CHECK-NEXT:   %[[Y:.+]] = arith.mulf %[[IF1]]#1, %arg3
//  CHECK-NEXT:   tt.store %[[XS]], %[[Y]]
// CANON-LABEL: @layer_norm_chain
//       CANON: scf.if
//   CANON-NOT: scf.if
//       CANON: tt.store %arg1
//  CANON-NEXT: tt.store %arg0
//       CANON: tt.store
//       CANON: } else {
//       CANON: tt.store %arg1
//  CANON-NEXT: tt.store %arg0
//       CANON: tt.store
// Over the cap the ops between the ifs stay put; CSE still merges the guards.
// CAP-LABEL: @layer_norm_chain
//       CAP: %[[G:.+]] = arith.cmpi
//       CAP: scf.if %[[G]]
//       CAP: "tt.reduce"
//       CAP: arith.sitofp
//  CAP-NEXT: arith.divf
//  CAP-NEXT: tt.splat
//  CAP-NEXT: scf.if %[[G]]
//       CAP: "tt.reduce"
//       CAP: tt.store
//       CAP: scf.if %[[G]]
tt.func public @layer_norm_chain(%X: !tt.ptr<f32>, %Mean: !tt.ptr<f32>, %N: i32, %a: tensor<128xf32>, %b: tensor<128xf32>) {
  %c0 = arith.constant 0 : i64
  %n = arith.extsi %N : i32 to i64
  %s = arith.addi %c0, %n : i64
  %c128 = arith.constant 128 : i64
  %r = arith.remsi %s, %c128 : i64
  %z = arith.constant 0 : i64
  %g = arith.cmpi eq, %r, %z : i64
  %0 = scf.if %g -> (tensor<128xf32>) {
    %t = arith.addf %a, %b : tensor<128xf32>
    scf.yield %t : tensor<128xf32>
  } else {
    %e = arith.subf %a, %b : tensor<128xf32>
    scf.yield %e : tensor<128xf32>
  }
  %sum = "tt.reduce"(%0) <{axis = 0 : i32}> ({
  ^bb0(%l: f32, %rr: f32):
    %x = arith.addf %l, %rr : f32
    tt.reduce.return %x : f32
  }) : (tensor<128xf32>) -> f32
  %nf = arith.sitofp %N : i32 to f32
  %mean = arith.divf %sum, %nf : f32
  %meanv = tt.splat %mean : f32 -> tensor<128xf32>
  %c0_2 = arith.constant 0 : i64
  %n_2 = arith.extsi %N : i32 to i64
  %s_2 = arith.addi %c0_2, %n_2 : i64
  %c128_2 = arith.constant 128 : i64
  %r_2 = arith.remsi %s_2, %c128_2 : i64
  %z_2 = arith.constant 0 : i64
  %g_2 = arith.cmpi eq, %r_2, %z_2 : i64
  %1 = scf.if %g_2 -> (tensor<128xf32>) {
    %t = arith.mulf %meanv, %a : tensor<128xf32>
    scf.yield %t : tensor<128xf32>
  } else {
    %e = arith.mulf %meanv, %b : tensor<128xf32>
    scf.yield %e : tensor<128xf32>
  }
  %sum2 = "tt.reduce"(%1) <{axis = 0 : i32}> ({
  ^bb0(%l: f32, %rr: f32):
    %x = arith.addf %l, %rr : f32
    tt.reduce.return %x : f32
  }) : (tensor<128xf32>) -> f32
  %var = arith.divf %sum2, %nf : f32
  tt.store %Mean, %mean : !tt.ptr<f32>
  tt.store %X, %var : !tt.ptr<f32>
  %varv = tt.splat %var : f32 -> tensor<128xf32>
  %c0_3 = arith.constant 0 : i64
  %n_3 = arith.extsi %N : i32 to i64
  %s_3 = arith.addi %c0_3, %n_3 : i64
  %c128_3 = arith.constant 128 : i64
  %r_3 = arith.remsi %s_3, %c128_3 : i64
  %z_3 = arith.constant 0 : i64
  %g_3 = arith.cmpi eq, %r_3, %z_3 : i64
  %xs = tt.splat %X : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  scf.if %g_3 {
    %y = arith.mulf %varv, %a : tensor<128xf32>
    tt.store %xs, %y : tensor<128x!tt.ptr<f32>>
  } else {
    %y = arith.mulf %varv, %b : tensor<128xf32>
    tt.store %xs, %y : tensor<128x!tt.ptr<f32>>
  }
  tt.return
}

// -----

// The op between the ifs uses the first if's result: each branch's copy must
// use that branch's own yielded value. The returned %m becomes an extra result,
// and the first if's own result is redirected to the rebuilt if.
// CHECK-LABEL: @sink_maps_results_per_branch
//       CHECK: %[[IF:.+]]:2 = scf.if %arg0 -> (f32, f32) {
//  CHECK-NEXT:   %[[T:.+]] = arith.addf %arg1, %arg2 : f32
//  CHECK-NEXT:   %[[TM:.+]] = arith.mulf %[[T]], %arg1 : f32
//  CHECK-NEXT:   tt.store %arg3, %[[TM]]
//  CHECK-NEXT:   scf.yield %[[T]], %[[TM]] : f32, f32
//  CHECK-NEXT: } else {
//  CHECK-NEXT:   %[[E:.+]] = arith.subf %arg1, %arg2 : f32
//  CHECK-NEXT:   %[[EM:.+]] = arith.mulf %[[E]], %arg1 : f32
//  CHECK-NEXT:   tt.store %arg3, %[[EM]]
//  CHECK-NEXT:   scf.yield %[[E]], %[[EM]] : f32, f32
//  CHECK-NEXT: }
//  CHECK-NEXT: scf.if %arg0 {
//  CHECK-NEXT:   tt.store %arg3, %[[IF]]#1
//       CHECK: tt.return %[[IF]]#0, %[[IF]]#1 : f32, f32
// CANON-LABEL: @sink_maps_results_per_branch
//       CANON: scf.if
//   CANON-NOT: scf.if
tt.func public @sink_maps_results_per_branch(%c: i1, %x: f32, %y: f32, %p: !tt.ptr<f32>) -> (f32, f32) {
  %0 = scf.if %c -> (f32) {
    %t = arith.addf %x, %y : f32
    scf.yield %t : f32
  } else {
    %e = arith.subf %x, %y : f32
    scf.yield %e : f32
  }
  %m = arith.mulf %0, %x : f32
  tt.store %p, %m : !tt.ptr<f32>
  scf.if %c {
    tt.store %p, %m : !tt.ptr<f32>
  } else {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.return %0, %m : f32, f32
}

// -----

// Pure ops in both gaps of a three-if chain all go before the first if.
// CHECK-LABEL: @three_ifs_pure_chain
//       CHECK: %[[M1:.+]] = arith.mulf %arg2, %arg3
//  CHECK-NEXT: %[[M2:.+]] = arith.addf %arg3, %arg2
//  CHECK-NEXT: scf.if %arg0
// CANON-LABEL: @three_ifs_pure_chain
//       CANON: scf.if
//   CANON-NOT: scf.if
tt.func public @three_ifs_pure_chain(%c: i1, %p: !tt.ptr<f32>, %x: f32, %y: f32) {
  %a = scf.if %c -> (f32) {
    tt.store %p, %x : !tt.ptr<f32>
    scf.yield %x : f32
  } else {
    tt.store %p, %y : !tt.ptr<f32>
    scf.yield %y : f32
  }
  %m1 = arith.mulf %x, %y : f32
  %b = scf.if %c -> (f32) {
    tt.store %p, %m1 : !tt.ptr<f32>
    scf.yield %m1 : f32
  } else {
    tt.store %p, %x : !tt.ptr<f32>
    scf.yield %x : f32
  }
  %m2 = arith.addf %y, %x : f32
  scf.if %c {
    tt.store %p, %m2 : !tt.ptr<f32>
  } else {
    tt.store %p, %y : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Inside a loop body, the copied op uses a value defined outside the loop.
// CHECK-LABEL: @sink_in_loop_uses_outer_value
//       CHECK: %[[OUTER:.+]] = arith.mulf %arg2, %arg2
//       CHECK: scf.for
//       CHECK:   scf.if %arg0
//  CHECK-NEXT:     tt.store %arg1, %arg2
//  CHECK-NEXT:     %[[A:.+]] = arith.addf %arg2, %[[OUTER]]
//  CHECK-NEXT:     tt.store %arg1, %[[A]]
//       CHECK:   } else {
//  CHECK-NEXT:     %[[B:.+]] = arith.addf %[[OUTER]], %[[OUTER]]
//  CHECK-NEXT:     tt.store %arg1, %[[B]]
// CANON-LABEL: @sink_in_loop_uses_outer_value
//       CANON: scf.for
//       CANON:   scf.if
//   CANON-NOT:   scf.if
tt.func public @sink_in_loop_uses_outer_value(%c: i1, %p: !tt.ptr<f32>, %x: f32, %n: index) {
  %outer = arith.mulf %x, %x : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = scf.if %c -> (f32) {
      tt.store %p, %x : !tt.ptr<f32>
      scf.yield %x : f32
    } else {
      scf.yield %outer : f32
    }
    %m = arith.addf %0, %outer : f32
    tt.store %p, %m : !tt.ptr<f32>
    scf.if %c {
      tt.store %p, %outer : !tt.ptr<f32>
    } else {
      tt.store %p, %x : !tt.ptr<f32>
    }
  }
  tt.return
}

// -----

// Inside a loop body, a pure op that uses a function argument is hoisted above
// the first if (arguments of enclosing blocks are available).
// CHECK-LABEL: @hoist_in_loop_uses_function_argument
//       CHECK: scf.for
//  CHECK-NEXT:   arith.mulf %arg2, %arg2
//  CHECK-NEXT:   scf.if %arg0
// CANON-LABEL: @hoist_in_loop_uses_function_argument
//       CANON: scf.for
//       CANON:   scf.if
//   CANON-NOT:   scf.if
tt.func public @hoist_in_loop_uses_function_argument(%c: i1, %p: !tt.ptr<f32>, %x: f32, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scf.if %c {
      tt.store %p, %x : !tt.ptr<f32>
    } else {
      tt.store %p, %x : !tt.ptr<f32>
    }
    %m = arith.mulf %x, %x : f32
    scf.if %c {
      tt.store %p, %m : !tt.ptr<f32>
    } else {
      tt.store %p, %x : !tt.ptr<f32>
    }
  }
  tt.return
}

// -----

// A pure loop between the ifs whose body (not its operands) uses the first if's
// result cannot be hoisted above that if; it is copied into both branches.
// CHECK-LABEL: @pure_loop_body_uses_first_result
//       CHECK: %[[IF:.+]]:2 = scf.if %arg0 -> (f32, f32) {
//       CHECK:   scf.for
//       CHECK: } else {
//       CHECK:   scf.for
//       CHECK: scf.if %arg0 {
//  CHECK-NEXT:   tt.store %arg1, %[[IF]]#1
// CANON-LABEL: @pure_loop_body_uses_first_result
//       CANON: scf.if
//   CANON-NOT: scf.if
tt.func public @pure_loop_body_uses_first_result(%c: i1, %p: !tt.ptr<f32>, %x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.if %c -> (f32) {
    %t = arith.addf %x, %x : f32
    scf.yield %t : f32
  } else {
    scf.yield %x : f32
  }
  %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %x) -> (f32) {
    %s = arith.addf %a, %0 : f32
    scf.yield %s : f32
  }
  scf.if %c {
    tt.store %p, %acc : !tt.ptr<f32>
  } else {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.return
}

// -----

// The second guard's remainder is also returned: after CSE the return uses the
// first remainder and both ifs test the first compare.
// CHECK-LABEL: @guard_value_with_other_use
//       CHECK: %[[R:.+]] = arith.remsi
//  CHECK-NEXT: %[[G:.+]] = arith.cmpi eq, %[[R]]
//       CHECK: scf.if %[[G]]
//       CHECK: scf.if %[[G]]
//       CHECK: tt.return %[[R]] : i64
// CANON-LABEL: @guard_value_with_other_use
//       CANON: scf.if
//   CANON-NOT: scf.if
tt.func public @guard_value_with_other_use(%N: i32, %p: !tt.ptr<f32>, %x: f32) -> i64 {
  %n = arith.extsi %N : i32 to i64
  %c128 = arith.constant 128 : i64
  %z = arith.constant 0 : i64
  %r = arith.remsi %n, %c128 : i64
  %g = arith.cmpi eq, %r, %z : i64
  %0 = scf.if %g -> (f32) {
    %t = arith.addf %x, %x : f32
    scf.yield %t : f32
  } else {
    scf.yield %x : f32
  }
  tt.store %p, %0 : !tt.ptr<f32>
  %n2 = arith.extsi %N : i32 to i64
  %c128_2 = arith.constant 128 : i64
  %z2 = arith.constant 0 : i64
  %r2 = arith.remsi %n2, %c128_2 : i64
  %g2 = arith.cmpi eq, %r2, %z2 : i64
  scf.if %g2 {
    tt.store %p, %x : !tt.ptr<f32>
  } else {
    tt.store %p, %0 : !tt.ptr<f32>
  }
  tt.return %r2 : i64
}

// -----

// Negative: the guards differ only in a constant (2048 vs 1024).
// CHECK-LABEL: @guard_constant_differs
//       CHECK: %[[G1:.+]] = arith.cmpi
//       CHECK: scf.if %[[G1]]
//       CHECK: tt.store
//       CHECK: %[[G2:.+]] = arith.cmpi
//       CHECK: scf.if %[[G2]]
// CANON-LABEL: @guard_constant_differs
//       CANON: scf.if
//       CANON: scf.if
tt.func public @guard_constant_differs(%N: i32, %p: !tt.ptr<f32>, %x: f32) {
  %n = arith.extsi %N : i32 to i64
  %c2048 = arith.constant 2048 : i64
  %z = arith.constant 0 : i64
  %r = arith.remsi %n, %c2048 : i64
  %g = arith.cmpi eq, %r, %z : i64
  %0 = scf.if %g -> (f32) {
    %t = arith.addf %x, %x : f32
    scf.yield %t : f32
  } else {
    scf.yield %x : f32
  }
  tt.store %p, %0 : !tt.ptr<f32>
  %c1024 = arith.constant 1024 : i64
  %r2 = arith.remsi %n, %c1024 : i64
  %g2 = arith.cmpi eq, %r2, %z : i64
  scf.if %g2 {
    tt.store %p, %x : !tt.ptr<f32>
  } else {
    tt.store %p, %0 : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Negative: guards computed from loads are never treated as the same condition;
// here a store between the loads can make them differ.
// CHECK-LABEL: @guards_from_loads
//       CHECK: scf.if %[[G1:.+]] {
//       CHECK: arith.mulf
//  CHECK-NEXT: scf.if %[[G2:.+]] {
// CANON-LABEL: @guards_from_loads
//       CANON: scf.if
//       CANON: scf.if
tt.func public @guards_from_loads(%q: !tt.ptr<i32>, %p: !tt.ptr<f32>, %x: f32) {
  %z = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %v = tt.load %q : !tt.ptr<i32>
  tt.store %q, %one : !tt.ptr<i32>
  %v2 = tt.load %q : !tt.ptr<i32>
  %g = arith.cmpi eq, %v, %z : i32
  %g2 = arith.cmpi eq, %v2, %z : i32
  scf.if %g {
    tt.store %p, %x : !tt.ptr<f32>
  } else {
    tt.store %p, %x : !tt.ptr<f32>
  }
  %m = arith.mulf %x, %x : f32
  scf.if %g2 {
    tt.store %p, %m : !tt.ptr<f32>
  } else {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Negative: the op between the ifs contains an scf.if.
// CHECK-LABEL: @nested_if_between
//       CHECK: scf.if %arg0
//       CHECK: scf.for
//       CHECK:   scf.if %arg1
//       CHECK: scf.if %arg0
// CANON-LABEL: @nested_if_between
//       CANON: scf.if %arg0
//       CANON: scf.for
//       CANON: scf.if %arg0
//  CANON-NEXT:   tt.store %arg2, %arg3
tt.func public @nested_if_between(%c: i1, %d: i1, %p: !tt.ptr<f32>, %x: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = scf.if %c -> (f32) {
    %t = arith.addf %x, %x : f32
    scf.yield %t : f32
  } else {
    scf.yield %x : f32
  }
  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %d {
      tt.store %p, %0 : !tt.ptr<f32>
    }
  }
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  } else {
    tt.store %p, %0 : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Negative: a tensor descriptor would have to leave the first if as a result.
// CHECK-LABEL: @tensordesc_would_escape
//       CHECK: scf.if %arg0 -> (i32) {
//       CHECK: } else {
//  CHECK-NEXT:   scf.yield
//  CHECK-NEXT: }
//  CHECK-NEXT: arith.extsi
//  CHECK-NEXT: tt.make_tensor_descriptor
//  CHECK-NEXT: scf.if %arg0 {
//  CHECK-NEXT:   tt.descriptor_load
tt.func public @tensordesc_would_escape(%c: i1, %base: !tt.ptr<f32>, %p: !tt.ptr<f32>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_f = arith.constant 1.000000e+00 : f32
  %0 = scf.if %c -> (i32) {
    tt.store %p, %c1_f : !tt.ptr<f32>
    scf.yield %c64_i32 : i32
  } else {
    scf.yield %c8_i32 : i32
  }
  %stride = arith.extsi %0 : i32 to i64
  %d = tt.make_tensor_descriptor %base, [%0, %0], [%stride, %c1_i64] : <f32>, <8x8xf32>
  scf.if %c {
    %v = tt.descriptor_load %d[%c0_i32, %c0_i32] : !tt.tensordesc<8x8xf32> -> tensor<8x8xf32>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    tt.store %ps, %v : tensor<8x8x!tt.ptr<f32>>
  } else {
    tt.store %p, %c1_f : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Negative: the first if has no else region and a non-hoistable op follows it.
// CHECK-LABEL: @no_else_blocking_op
//       CHECK: scf.if %arg0 {
//       CHECK: }
//  CHECK-NEXT: tt.store
//  CHECK-NEXT: scf.if %arg0
tt.func public @no_else_blocking_op(%c: i1, %p: !tt.ptr<f32>, %x: f32) {
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.store %p, %x : !tt.ptr<f32>
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Without an else region, a hoistable op between the ifs is still hoisted.
// CHECK-LABEL: @no_else_hoistable_op
//       CHECK: arith.mulf
//  CHECK-NEXT: scf.if %arg0
// CANON-LABEL: @no_else_hoistable_op
//       CANON: scf.if
//   CANON-NOT: scf.if
tt.func public @no_else_hoistable_op(%c: i1, %p: !tt.ptr<f32>, %x: f32) {
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  }
  %m = arith.mulf %x, %x : f32
  scf.if %c {
    tt.store %p, %m : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Conservative rejection: opaque pure inline assembly is effect-free but not on
// the uniform whitelist. The tied constraint makes it an identity on its input.
// CHECK-LABEL: @inline_asm_condition
//       CHECK: scf.if
//       CHECK: "tt.reduce"
//       CHECK: scf.if
// CANON-LABEL: @inline_asm_condition
//       CANON: scf.if
//       CANON: "tt.reduce"
//       CANON: scf.if
tt.func public @inline_asm_condition(%p: !tt.ptr<f32>, %a: tensor<64xf32>) {
  %z = arith.constant 0 : i32
  %v = tt.elementwise_inline_asm "" {constraints = "=r,0", packed_element = 1 : i32, pure = true} %z : i32 -> i32
  %c = arith.cmpi eq, %v, %z : i32
  %0 = scf.if %c -> (tensor<64xf32>) {
    scf.yield %a : tensor<64xf32>
  } else {
    %n = arith.negf %a : tensor<64xf32>
    scf.yield %n : tensor<64xf32>
  }
  %s = "tt.reduce"(%0) <{axis = 0 : i32}> ({
  ^bb0(%l: f32, %r: f32):
    %x = arith.addf %l, %r : f32
    tt.reduce.return %x : f32
  }) : (tensor<64xf32>) -> f32
  scf.if %c {
    tt.store %p, %s : !tt.ptr<f32>
  } else {
    tt.store %p, %s : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Negative: in a private helper the condition argument may come from a caller
// that computed it per warp.
// CHECK-LABEL: @private_helper_condition
//       CHECK: scf.if
//       CHECK: "tt.reduce"
//       CHECK: scf.if
// CANON-LABEL: @private_helper_condition
//       CANON: scf.if
//       CANON: "tt.reduce"
//       CANON: scf.if
tt.func private @private_helper_condition(%c: i1, %p: !tt.ptr<f32>, %a: tensor<64xf32>) {
  %0 = scf.if %c -> (tensor<64xf32>) {
    scf.yield %a : tensor<64xf32>
  } else {
    %n = arith.negf %a : tensor<64xf32>
    scf.yield %n : tensor<64xf32>
  }
  %s = "tt.reduce"(%0) <{axis = 0 : i32}> ({
  ^bb0(%l: f32, %r: f32):
    %x = arith.addf %l, %r : f32
    tt.reduce.return %x : f32
  }) : (tensor<64xf32>) -> f32
  scf.if %c {
    tt.store %p, %s : !tt.ptr<f32>
  } else {
    tt.store %p, %s : !tt.ptr<f32>
  }
  tt.return
}

// -----

// Shaped like Inductor's Welford layer_norm forward: a 3-result if feeds a
// 3-operand reduction, and the gap also holds two masked stores and an rsqrt.
// The two broadcasts are used independently by the next if's loop, so each
// branch must yield its own pair, in order, next to the original three results.
// CHECK-LABEL: @inductor_welford_layer_norm
//       CHECK: %[[G:.+]] = arith.cmpi slt
//       CHECK: %[[IF:.+]]:5 = scf.if %[[G]] ->
//       CHECK:   %[[A0:.+]]:3 = scf.for
//       CHECK:   "tt.reduce"(%[[A0]]#0, %[[A0]]#1, %[[A0]]#2)
//       CHECK:   tt.store %{{.+}}, %[[MEAN0:.+]], %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[RS0:.+]] = tt.extern_elementwise
//       CHECK:   tt.store %{{.+}}, %[[RS0]], %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[MB0:.+]] = tt.broadcast %[[MEAN0]]
//       CHECK:   %[[RB0:.+]] = tt.broadcast %[[RS0]]
//       CHECK:   scf.yield %[[A0]]#0, %[[A0]]#1, %[[A0]]#2, %[[MB0]], %[[RB0]]
//       CHECK: } else {
//       CHECK:   %[[A1:.+]]:3 = scf.for
//       CHECK:   "tt.reduce"(%[[A1]]#0, %[[A1]]#1, %[[A1]]#2)
//       CHECK:   tt.store %{{.+}}, %[[MEAN1:.+]], %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[RS1:.+]] = tt.extern_elementwise
//       CHECK:   tt.store %{{.+}}, %[[RS1]], %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[MB1:.+]] = tt.broadcast %[[MEAN1]]
//       CHECK:   %[[RB1:.+]] = tt.broadcast %[[RS1]]
//       CHECK:   scf.yield %[[A1]]#0, %[[A1]]#1, %[[A1]]#2, %[[MB1]], %[[RB1]]
//       CHECK: }
//  CHECK-NEXT: scf.if %[[G]] {
//       CHECK:   arith.subf %{{.+}}, %[[IF]]#3
//       CHECK:   arith.mulf %{{.+}}, %[[IF]]#4
// CANON-LABEL: @inductor_welford_layer_norm
//       CANON: scf.if
//       CANON:   "tt.reduce"
//       CANON: } else {
//       CANON:   "tt.reduce"
//   CANON-NOT: scf.if
//       CANON: tt.return
tt.func public @inductor_welford_layer_norm(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %mean_ptr: !tt.ptr<f32>, %rstd_ptr: !tt.ptr<f32>, %mask: tensor<1x128xi1>, %xnumel: i32, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %step = arith.constant 128 : i32
  %zero = arith.constant dense<0.000000e+00> : tensor<1x128xf32>
  %pid = tt.get_program_id x : i32
  %g = arith.cmpi slt, %pid, %xnumel : i32
  %inp = tt.splat %in : !tt.ptr<f32> -> tensor<1x128x!tt.ptr<f32>>
  %outp = tt.splat %out : !tt.ptr<f32> -> tensor<1x128x!tt.ptr<f32>>
  %st:3 = scf.if %g -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) {
    %acc:3 = scf.for %off = %c0 to %rnumel step %step iter_args(%a = %zero, %b = %zero, %c = %zero) -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) : i32 {
      %x = tt.load %inp : tensor<1x128x!tt.ptr<f32>>
      %a1 = arith.addf %a, %x : tensor<1x128xf32>
      %b1 = arith.mulf %b, %x : tensor<1x128xf32>
      %c1 = arith.subf %c, %x : tensor<1x128xf32>
      scf.yield %a1, %b1, %c1 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>
    }
    scf.yield %acc#0, %acc#1, %acc#2 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>
  } else {
    %acc:3 = scf.for %off = %c0 to %rnumel step %step iter_args(%a = %zero, %b = %zero, %c = %zero) -> (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) : i32 {
      %x = tt.load %inp, %mask, %zero : tensor<1x128x!tt.ptr<f32>>
      %a1 = arith.addf %a, %x : tensor<1x128xf32>
      %b1 = arith.mulf %b, %x : tensor<1x128xf32>
      %c1 = arith.subf %c, %x : tensor<1x128xf32>
      scf.yield %a1, %b1, %c1 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>
    }
    scf.yield %acc#0, %acc#1, %acc#2 : tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>
  }
  %wf:3 = "tt.reduce"(%st#0, %st#1, %st#2) <{axis = 1 : i32}> ({
  ^bb0(%a0: f32, %a1: f32, %a2: f32, %b0: f32, %b1: f32, %b2: f32):
    %s0 = arith.addf %a0, %b0 : f32
    %s1 = arith.addf %a1, %b1 : f32
    %s2 = arith.addf %a2, %b2 : f32
    tt.reduce.return %s0, %s1, %s2 : f32, f32, f32
  }) : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>) -> (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
  %mean = tt.expand_dims %wf#0 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
  %m2v = tt.expand_dims %wf#1 {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
  %mptr = tt.splat %mean_ptr : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
  %xm = tt.splat %g : i1 -> tensor<1x1xi1>
  tt.store %mptr, %mean, %xm : tensor<1x1x!tt.ptr<f32>>
  %nf = arith.sitofp %rnumel : i32 to f32
  %nfv = tt.splat %nf : f32 -> tensor<1x1xf32>
  %var = arith.divf %m2v, %nfv : tensor<1x1xf32>
  %rstd = tt.extern_elementwise %var {libname = "", libpath = "", pure = true, symbol = "__imf_rsqrtf"} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %rptr = tt.splat %rstd_ptr : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
  tt.store %rptr, %rstd, %xm : tensor<1x1x!tt.ptr<f32>>
  %meanb = tt.broadcast %mean : tensor<1x1xf32> -> tensor<1x128xf32>
  %rstdb = tt.broadcast %rstd : tensor<1x1xf32> -> tensor<1x128xf32>
  scf.if %g {
    scf.for %off = %c0 to %rnumel step %step : i32 {
      %x = tt.load %inp : tensor<1x128x!tt.ptr<f32>>
      %c = arith.subf %x, %meanb : tensor<1x128xf32>
      %y = arith.mulf %c, %rstdb : tensor<1x128xf32>
      tt.store %outp, %y : tensor<1x128x!tt.ptr<f32>>
    }
  } else {
    scf.for %off = %c0 to %rnumel step %step : i32 {
      %x = tt.load %inp, %mask, %zero : tensor<1x128x!tt.ptr<f32>>
      %c = arith.subf %x, %meanb : tensor<1x128xf32>
      %y = arith.mulf %c, %rstdb : tensor<1x128xf32>
      tt.store %outp, %y, %mask : tensor<1x128x!tt.ptr<f32>>
    }
  }
  tt.return
}

// -----

// The rsqrt depends on the first if's result, so the gap is copied into both
// branches. The copied gap is 8 ops (the reduction with its two region ops,
// expand_dims, divf, rsqrt, store and broadcast); hoisted ops do not count. One
// op less keeps the ifs apart.
// CHECK-LABEL: @clone_cap_boundary
//       CHECK: %[[G:.+]] = arith.cmpi slt
//       CHECK: arith.sitofp
//       CHECK: %[[IF:.+]]:2 = scf.if %[[G]] ->
//       CHECK:   "tt.reduce"
//       CHECK:   tt.extern_elementwise
//       CHECK:   tt.store %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[B0:.+]] = tt.broadcast
//       CHECK:   scf.yield %{{.+}}, %[[B0]]
//       CHECK: } else {
//       CHECK:   "tt.reduce"
//       CHECK:   tt.extern_elementwise
//       CHECK:   tt.store %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x1x!tt.ptr<f32>>
//       CHECK:   %[[B1:.+]] = tt.broadcast
//       CHECK:   scf.yield %{{.+}}, %[[B1]]
//       CHECK: }
//  CHECK-NEXT: scf.if %[[G]] {
//       CHECK:   arith.mulf %{{.+}}, %[[IF]]#1
// CANON-LABEL: @clone_cap_boundary
//       CANON: scf.if
//       CANON:   "tt.reduce"
//       CANON: } else {
//       CANON:   "tt.reduce"
//   CANON-NOT: scf.if
//       CANON: tt.return
// BELOW-LABEL: @clone_cap_boundary
//       BELOW: scf.if
//       BELOW: "tt.reduce"
//       BELOW: tt.broadcast
//  BELOW-NEXT: scf.if
//    AT-LABEL: @clone_cap_boundary
//          AT: scf.if
//          AT: "tt.reduce"
//      AT-NOT: scf.if
//          AT: tt.return
tt.func public @clone_cap_boundary(%a: tensor<1x128xf32>, %rstd: !tt.ptr<f32>, %out: !tt.ptr<f32>, %xnumel: i32, %rnumel: i32) {
  %pid = tt.get_program_id x : i32
  %g = arith.cmpi slt, %pid, %xnumel : i32
  %outp = tt.splat %out : !tt.ptr<f32> -> tensor<1x128x!tt.ptr<f32>>
  %sumsq = scf.if %g -> (tensor<1x128xf32>) {
    scf.yield %a : tensor<1x128xf32>
  } else {
    %n = arith.negf %a : tensor<1x128xf32>
    scf.yield %n : tensor<1x128xf32>
  }
  %sum = "tt.reduce"(%sumsq) <{axis = 1 : i32}> ({
  ^bb0(%l: f32, %rr: f32):
    %add = arith.addf %l, %rr : f32
    tt.reduce.return %add : f32
  }) : (tensor<1x128xf32>) -> tensor<1xf32>
  %sum2 = tt.expand_dims %sum {axis = 1 : i32} : tensor<1xf32> -> tensor<1x1xf32>
  %nf = arith.sitofp %rnumel : i32 to f32
  %nfv = tt.splat %nf : f32 -> tensor<1x1xf32>
  %mean = arith.divf %sum2, %nfv : tensor<1x1xf32>
  %rs = tt.extern_elementwise %mean {libname = "", libpath = "", pure = true, symbol = "__imf_rsqrtf"} : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %rptr = tt.splat %rstd : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
  %xm = tt.splat %g : i1 -> tensor<1x1xi1>
  tt.store %rptr, %rs, %xm : tensor<1x1x!tt.ptr<f32>>
  %rsb = tt.broadcast %rs : tensor<1x1xf32> -> tensor<1x128xf32>
  scf.if %g {
    %y = arith.mulf %a, %rsb : tensor<1x128xf32>
    tt.store %outp, %y : tensor<1x128x!tt.ptr<f32>>
  } else {
    %n = arith.negf %a : tensor<1x128xf32>
    %y = arith.mulf %n, %rsb : tensor<1x128xf32>
    tt.store %outp, %y : tensor<1x128x!tt.ptr<f32>>
  }
  tt.return
}

// -----

// The uniformity check accepts tt.get_num_programs. It only runs when
// something has to be copied, so the gap holds a reduction that depends on the
// first if's result and the later store uses it.
// CHECK-LABEL: @guard_with_num_programs
//       CHECK: %[[C:.+]] = arith.cmpi sgt
//       CHECK: %[[IF:.+]]:2 = scf.if %[[C]] -> (tensor<64xf32>, f32) {
//       CHECK:   %[[S0:.+]] = "tt.reduce"
//       CHECK:   scf.yield %{{.+}}, %[[S0]]
//       CHECK: } else {
//       CHECK:   %[[S1:.+]] = "tt.reduce"
//       CHECK:   scf.yield %{{.+}}, %[[S1]]
//       CHECK: }
//  CHECK-NEXT: scf.if %[[C]] {
//  CHECK-NEXT:   tt.store %arg0, %[[IF]]#1
// CANON-LABEL: @guard_with_num_programs
//       CANON: scf.if
//   CANON-NOT: scf.if
//       CANON: tt.return
tt.func public @guard_with_num_programs(%p: !tt.ptr<f32>, %a: tensor<64xf32>, %n: i32) {
  %np = tt.get_num_programs x : i32
  %c = arith.cmpi sgt, %np, %n : i32
  %0 = scf.if %c -> (tensor<64xf32>) {
    scf.yield %a : tensor<64xf32>
  } else {
    %neg = arith.negf %a : tensor<64xf32>
    scf.yield %neg : tensor<64xf32>
  }
  %s = "tt.reduce"(%0) <{axis = 0 : i32}> ({
  ^bb0(%l: f32, %r: f32):
    %x = arith.addf %l, %r : f32
    tt.reduce.return %x : f32
  }) : (tensor<64xf32>) -> f32
  scf.if %c {
    tt.store %p, %s : !tt.ptr<f32>
  } else {
    tt.store %p, %s : !tt.ptr<f32>
  }
  tt.return
}

// -----

// CSE erases the dead middle if, so the ifs must be collected after CSE.
// CHECK-LABEL: @dead_if_between
//   CHECK-NOT: arith.addf
tt.func public @dead_if_between(%c: i1, %p: !tt.ptr<f32>, %x: f32) {
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  }
  scf.if %c {
    %t = arith.addf %x, %x : f32
  }
  scf.if %c {
    tt.store %p, %x : !tt.ptr<f32>
  }
  tt.return
}
