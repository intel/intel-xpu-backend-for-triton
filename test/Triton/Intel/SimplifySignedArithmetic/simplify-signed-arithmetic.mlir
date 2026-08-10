// RUN: triton-opt %s -split-input-file -triton-intel-simplify-signed-arithmetic | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive Tests - Should convert signed to unsigned
//===----------------------------------------------------------------------===//

// Test 1: get_program_id is non-negative, constant divisor is positive -> convert remsi to remui
module {
tt.func public @remsi_from_program_id() -> i32 {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %pid, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_program_id
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[REM:.*]] = arith.remui %[[PID]], %[[C128]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 2: make_range with start=0 is non-negative -> convert
module {
tt.func public @remsi_from_make_range() -> tensor<128xi32> {
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %c64 = arith.constant dense<64> : tensor<128xi32>
  %rem = arith.remsi %range, %c64 : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @remsi_from_make_range
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[C64:.*]] = arith.constant dense<64>
// CHECK: %[[REM:.*]] = arith.remui %[[RANGE]], %[[C64]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 3: Non-negative constant dividend -> convert
module {
tt.func public @remsi_from_constant() -> i32 {
  %c100 = arith.constant 100 : i32
  %c7 = arith.constant 7 : i32
  %rem = arith.remsi %c100, %c7 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_constant
// CHECK: %[[C100:.*]] = arith.constant 100
// CHECK: %[[C7:.*]] = arith.constant 7
// CHECK: %[[REM:.*]] = arith.remui %[[C100]], %[[C7]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 4: addi of two non-negative values -> convert
module {
tt.func public @remsi_from_addi() -> i32 {
  %pid = tt.get_program_id x : i32
  %c10 = arith.constant 10 : i32
  %sum = arith.addi %pid, %c10 : i32
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %sum, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_addi
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C10:.*]] = arith.constant 10
// CHECK: %[[SUM:.*]] = arith.addi %[[PID]], %[[C10]]
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[REM:.*]] = arith.remui %[[SUM]], %[[C128]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 5: muli of two non-negative values -> convert
module {
tt.func public @remsi_from_muli() -> i32 {
  %pid = tt.get_program_id x : i32
  %c4 = arith.constant 4 : i32
  %prod = arith.muli %pid, %c4 : i32
  %c256 = arith.constant 256 : i32
  %rem = arith.remsi %prod, %c256 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_muli
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C4:.*]] = arith.constant 4
// CHECK: %[[PROD:.*]] = arith.muli %[[PID]], %[[C4]]
// CHECK: %[[C256:.*]] = arith.constant 256
// CHECK: %[[REM:.*]] = arith.remui %[[PROD]], %[[C256]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 6: get_num_programs is non-negative -> convert
module {
tt.func public @remsi_from_num_programs() -> i32 {
  %np = tt.get_num_programs x : i32
  %c32 = arith.constant 32 : i32
  %rem = arith.remsi %np, %c32 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_num_programs
// CHECK: %[[NP:.*]] = tt.get_num_programs x
// CHECK: %[[C32:.*]] = arith.constant 32
// CHECK: %[[REM:.*]] = arith.remui %[[NP]], %[[C32]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 7: splat of non-negative scalar -> convert
module {
tt.func public @remsi_from_splat() -> tensor<128xi32> {
  %pid = tt.get_program_id x : i32
  %splat = tt.splat %pid : i32 -> tensor<128xi32>
  %c64 = arith.constant dense<64> : tensor<128xi32>
  %rem = arith.remsi %splat, %c64 : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @remsi_from_splat
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[SPLAT:.*]] = tt.splat %[[PID]]
// CHECK: %[[C64:.*]] = arith.constant dense<64>
// CHECK: %[[REM:.*]] = arith.remui %[[SPLAT]], %[[C64]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 8: expand_dims preserves non-negativity -> convert
module {
tt.func public @remsi_from_expand_dims() -> tensor<1x128xi32> {
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %c32 = arith.constant dense<32> : tensor<1x128xi32>
  %rem = arith.remsi %expanded, %c32 : tensor<1x128xi32>
  tt.return %rem : tensor<1x128xi32>
}
// CHECK-LABEL: @remsi_from_expand_dims
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[RANGE]]
// CHECK: %[[C32:.*]] = arith.constant dense<32>
// CHECK: %[[REM:.*]] = arith.remui %[[EXPANDED]], %[[C32]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 9: broadcast preserves non-negativity -> convert
module {
tt.func public @remsi_from_broadcast() -> tensor<64x128xi32> {
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %broadcast = tt.broadcast %expanded : tensor<1x128xi32> -> tensor<64x128xi32>
  %c16 = arith.constant dense<16> : tensor<64x128xi32>
  %rem = arith.remsi %broadcast, %c16 : tensor<64x128xi32>
  tt.return %rem : tensor<64x128xi32>
}
// CHECK-LABEL: @remsi_from_broadcast
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[RANGE]]
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]]
// CHECK: %[[C16:.*]] = arith.constant dense<16>
// CHECK: %[[REM:.*]] = arith.remui %[[BROADCAST]], %[[C16]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 10: andi with non-negative mask -> convert
module {
tt.func public @remsi_from_andi(%arg0: i32) -> i32 {
  %c0x7FFF = arith.constant 32767 : i32
  %masked = arith.andi %arg0, %c0x7FFF : i32
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %masked, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @remsi_from_andi
// CHECK: %[[MASK:.*]] = arith.constant 32767
// CHECK: %[[MASKED:.*]] = arith.andi %arg0, %[[MASK]]
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[REM:.*]] = arith.remui %[[MASKED]], %[[C128]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 11: Typical Triton indexing pattern - pid * BLOCK_SIZE + make_range
module {
tt.func public @typical_triton_indexing() -> tensor<128xi32> {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %offset = arith.muli %pid, %c128 : i32
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %offset_splat = tt.splat %offset : i32 -> tensor<128xi32>
  %idx = arith.addi %offset_splat, %range : tensor<128xi32>
  %c256 = arith.constant dense<256> : tensor<128xi32>
  %rem = arith.remsi %idx, %c256 : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @typical_triton_indexing
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[OFFSET:.*]] = arith.muli %[[PID]], %[[C128]]
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[SPLAT:.*]] = tt.splat %[[OFFSET]]
// CHECK: %[[IDX:.*]] = arith.addi %[[SPLAT]], %[[RANGE]]
// CHECK: %[[C256:.*]] = arith.constant dense<256>
// CHECK: %[[REM:.*]] = arith.remui %[[IDX]], %[[C256]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 12: divsi with non-negative dividend and positive divisor -> convert to divui
module {
tt.func public @divsi_from_program_id() -> i32 {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %pid, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @divsi_from_program_id
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[DIV:.*]] = arith.divui %[[PID]], %[[C128]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 13: Chained divsi/remsi pattern (y1 = idx / 128, y1_rem = y1 % 64)
module {
tt.func public @chained_div_rem() -> i32 {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %c64 = arith.constant 64 : i32
  %y1 = arith.divsi %pid, %c128 : i32
  %y1_rem = arith.remsi %y1, %c64 : i32
  tt.return %y1_rem : i32
}
// CHECK-LABEL: @chained_div_rem
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[C64:.*]] = arith.constant 64
// CHECK: %[[Y1:.*]] = arith.divui %[[PID]], %[[C128]]
// CHECK: %[[Y1_REM:.*]] = arith.remui %[[Y1]], %[[C64]]
// CHECK: tt.return %[[Y1_REM]]
}

// -----

// Test 14: divsi with tensor types
module {
tt.func public @divsi_tensor() -> tensor<128xi32> {
  %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %c32 = arith.constant dense<32> : tensor<128xi32>
  %div = arith.divsi %range, %c32 : tensor<128xi32>
  tt.return %div : tensor<128xi32>
}
// CHECK-LABEL: @divsi_tensor
// CHECK: %[[RANGE:.*]] = tt.make_range
// CHECK: %[[C32:.*]] = arith.constant dense<32>
// CHECK: %[[DIV:.*]] = arith.divui %[[RANGE]], %[[C32]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 15: Multiple chained divsi operations
module {
tt.func public @multiple_chained_divsi() -> i32 {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %c8192 = arith.constant 8192 : i32
  %y1 = arith.divsi %pid, %c128 : i32
  %y5 = arith.divsi %pid, %c8192 : i32
  %sum = arith.addi %y1, %y5 : i32
  tt.return %sum : i32
}
// CHECK-LABEL: @multiple_chained_divsi
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[C8192:.*]] = arith.constant 8192
// CHECK: %[[Y1:.*]] = arith.divui %[[PID]], %[[C128]]
// CHECK: %[[Y5:.*]] = arith.divui %[[PID]], %[[C8192]]
// CHECK: %[[SUM:.*]] = arith.addi %[[Y1]], %[[Y5]]
// CHECK: tt.return %[[SUM]]
}

// Test 16: remsi with non-negative dividend produces non-negative result
// even when divisor is negative (truncation toward zero preserves sign of
// dividend). The downstream remsi should be converted.
module {
tt.func public @remsi_nonneg_dividend_neg_divisor() -> i32 {
  %pid = tt.get_program_id x : i32
  %cn7 = arith.constant -7 : i32
  %rem = arith.remsi %pid, %cn7 : i32
  %c64 = arith.constant 64 : i32
  %rem2 = arith.remsi %rem, %c64 : i32
  tt.return %rem2 : i32
}
// CHECK-LABEL: @remsi_nonneg_dividend_neg_divisor
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[CN7:.*]] = arith.constant -7
// CHECK: %[[REM:.*]] = arith.remsi %[[PID]], %[[CN7]]
// CHECK: %[[C64:.*]] = arith.constant 64
// CHECK: %[[REM2:.*]] = arith.remui %[[REM]], %[[C64]]
// CHECK: tt.return %[[REM2]]
}

// -----

// Test 17: remsi with non-negative dividend and non-constant divisor
// produces non-negative result. Downstream divsi should be converted.
module {
tt.func public @remsi_nonneg_dividend_nonconst_divisor(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %rem = arith.remsi %pid, %arg0 : i32
  %c32 = arith.constant 32 : i32
  %div = arith.divsi %rem, %c32 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @remsi_nonneg_dividend_nonconst_divisor
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[REM:.*]] = arith.remsi %[[PID]], %arg0
// CHECK: %[[C32:.*]] = arith.constant 32
// CHECK: %[[DIV:.*]] = arith.divui %[[REM]], %[[C32]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 18: divsi with non-negative dividend and get_num_programs divisor.
// get_num_programs is strictly positive, so the divsi converts to divui; its
// non-negative result then lets the downstream remsi convert to remui.
module {
tt.func public @divsi_nonneg_both_operands() -> i32 {
  %pid = tt.get_program_id x : i32
  %np = tt.get_num_programs x : i32
  %div = arith.divsi %pid, %np : i32
  %c16 = arith.constant 16 : i32
  %rem = arith.remsi %div, %c16 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @divsi_nonneg_both_operands
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[NP:.*]] = tt.get_num_programs x
// CHECK: %[[DIV:.*]] = arith.divui %[[PID]], %[[NP]]
// CHECK: %[[C16:.*]] = arith.constant 16
// CHECK: %[[REM:.*]] = arith.remui %[[DIV]], %[[C16]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 19: maxsi with non-negative LHS produces non-negative result.
// Downstream divsi should be converted.
module {
tt.func public @maxsi_nonneg_lhs(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %max = arith.maxsi %pid, %arg0 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %max, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @maxsi_nonneg_lhs
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[MAX:.*]] = arith.maxsi %[[PID]], %arg0
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[DIV:.*]] = arith.divui %[[MAX]], %[[C128]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 20: maxsi with non-negative RHS produces non-negative result.
// Downstream remsi should be converted.
module {
tt.func public @maxsi_nonneg_rhs(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %max = arith.maxsi %arg0, %c0 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %max, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @maxsi_nonneg_rhs
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: %[[MAX:.*]] = arith.maxsi %arg0, %[[C0]]
// CHECK: %[[C64:.*]] = arith.constant 64
// CHECK: %[[REM:.*]] = arith.remui %[[MAX]], %[[C64]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 21: maxsi result used in chained computation.
// maxsi(arg, 0) is non-negative, addi with pid is non-negative, divsi converts.
module {
tt.func public @maxsi_in_chain(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %max = arith.maxsi %arg0, %c0 : i32
  %pid = tt.get_program_id x : i32
  %sum = arith.addi %max, %pid : i32
  %c256 = arith.constant 256 : i32
  %div = arith.divsi %sum, %c256 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @maxsi_in_chain
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: %[[MAX:.*]] = arith.maxsi %arg0, %[[C0]]
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[SUM:.*]] = arith.addi %[[MAX]], %[[PID]]
// CHECK: %[[C256:.*]] = arith.constant 256
// CHECK: %[[DIV:.*]] = arith.divui %[[SUM]], %[[C256]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 22: andi with non-negative first operand and non-constant second -> convert
module {
tt.func public @andi_nonneg_nonconstant_operand(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %m = arith.andi %pid, %arg0 : i32
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %m, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @andi_nonneg_nonconstant_operand
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[M:.*]] = arith.andi %[[PID]], %arg0
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[REM:.*]] = arith.remui %[[M]], %[[C128]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 23: extui always produces non-negative result -> convert
module {
tt.func public @extui_always_nonneg(%arg0: i16) -> i32 {
  %e = arith.extui %arg0 : i16 to i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %e, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @extui_always_nonneg
// CHECK: %[[E:.*]] = arith.extui %arg0
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[DIV:.*]] = arith.divui %[[E]], %[[C128]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 24: extsi with non-negative input -> convert
module {
tt.func public @extsi_nonneg_input() -> i32 {
  %c5 = arith.constant 5 : i16
  %e = arith.extsi %c5 : i16 to i32
  %c8 = arith.constant 8 : i32
  %div = arith.divsi %e, %c8 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @extsi_nonneg_input
// CHECK: %[[C5:.*]] = arith.constant 5
// CHECK: %[[E:.*]] = arith.extsi %[[C5]]
// CHECK: %[[C8:.*]] = arith.constant 8
// CHECK: %[[DIV:.*]] = arith.divui %[[E]], %[[C8]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 25: shrsi with non-negative LHS -> convert
module {
tt.func public @shrsi_nonneg_lhs() -> i32 {
  %pid = tt.get_program_id x : i32
  %c2 = arith.constant 2 : i32
  %s = arith.shrsi %pid, %c2 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %s, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @shrsi_nonneg_lhs
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C2:.*]] = arith.constant 2
// CHECK: %[[S:.*]] = arith.shrsi %[[PID]], %[[C2]]
// CHECK: %[[C64:.*]] = arith.constant 64
// CHECK: %[[REM:.*]] = arith.remui %[[S]], %[[C64]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 26: minsi with both operands non-negative -> convert
module {
tt.func public @minsi_both_nonneg() -> i32 {
  %pid = tt.get_program_id x : i32
  %c5 = arith.constant 5 : i32
  %m = arith.minsi %pid, %c5 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %m, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @minsi_both_nonneg
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C5:.*]] = arith.constant 5
// CHECK: %[[M:.*]] = arith.minsi %[[PID]], %[[C5]]
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[DIV:.*]] = arith.divui %[[M]], %[[C128]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 27: select with both operands non-negative -> convert
module {
tt.func public @select_both_nonneg(%cond: i1) -> i32 {
  %pid = tt.get_program_id x : i32
  %c5 = arith.constant 5 : i32
  %s = arith.select %cond, %pid, %c5 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %s, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @select_both_nonneg
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C5:.*]] = arith.constant 5
// CHECK: %[[S:.*]] = arith.select %arg0, %[[PID]], %[[C5]]
// CHECK: %[[C64:.*]] = arith.constant 64
// CHECK: %[[REM:.*]] = arith.remui %[[S]], %[[C64]]
// CHECK: tt.return %[[REM]]
}

// -----

// Test 28: divsi by get_num_programs (strictly positive) -> convert
module {
tt.func public @divsi_by_num_programs() -> i32 {
  %pid = tt.get_program_id x : i32
  %np = tt.get_num_programs x : i32
  %div = arith.divsi %pid, %np : i32
  tt.return %div : i32
}
// CHECK-LABEL: @divsi_by_num_programs
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[NP:.*]] = tt.get_num_programs x
// CHECK: %[[DIV:.*]] = arith.divui %[[PID]], %[[NP]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 29: divsi by maxsi with positive constant (strictly positive) -> convert
module {
tt.func public @divsi_by_maxsi_positive_const(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %c1 = arith.constant 1 : i32
  %mx = arith.maxsi %arg0, %c1 : i32
  %div = arith.divsi %pid, %mx : i32
  tt.return %div : i32
}
// CHECK-LABEL: @divsi_by_maxsi_positive_const
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C1:.*]] = arith.constant 1
// CHECK: %[[MX:.*]] = arith.maxsi %arg0, %[[C1]]
// CHECK: %[[DIV:.*]] = arith.divui %[[PID]], %[[MX]]
// CHECK: tt.return %[[DIV]]
}

// -----

// Test 30: ceildivsi with non-negative dividend and positive divisor -> convert to ceildivui
module {
tt.func public @ceildivsi_from_program_id() -> i32 {
  %pid = tt.get_program_id x : i32
  %c128 = arith.constant 128 : i32
  %cdiv = arith.ceildivsi %pid, %c128 : i32
  tt.return %cdiv : i32
}
// CHECK-LABEL: @ceildivsi_from_program_id
// CHECK: %[[PID:.*]] = tt.get_program_id x
// CHECK: %[[C128:.*]] = arith.constant 128
// CHECK: %[[CDIV:.*]] = arith.ceildivui %[[PID]], %[[C128]]
// CHECK: tt.return %[[CDIV]]
}

// -----

//===----------------------------------------------------------------------===//
// Negative Tests - Should NOT convert
//===----------------------------------------------------------------------===//

// -----

// Negative Test 1: Function argument (unknown sign) -> do NOT convert
module {
tt.func public @no_convert_unknown_arg(%arg0: i32) -> i32 {
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %arg0, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_unknown_arg
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 2: Negative constant dividend -> do NOT convert
module {
tt.func public @no_convert_negative_dividend() -> i32 {
  %cn10 = arith.constant -10 : i32
  %c7 = arith.constant 7 : i32
  %rem = arith.remsi %cn10, %c7 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_negative_dividend
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 3: Zero divisor -> do NOT convert (not positive)
module {
tt.func public @no_convert_zero_divisor() -> i32 {
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %rem = arith.remsi %pid, %c0 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_zero_divisor
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 4: Negative constant divisor -> do NOT convert
module {
tt.func public @no_convert_negative_divisor() -> i32 {
  %pid = tt.get_program_id x : i32
  %cn128 = arith.constant -128 : i32
  %rem = arith.remsi %pid, %cn128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_negative_divisor
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 5: make_range with negative start -> do NOT convert
module {
tt.func public @no_convert_negative_range() -> tensor<128xi32> {
  %range = tt.make_range {start = -64 : i32, end = 64 : i32} : tensor<128xi32>
  %c32 = arith.constant dense<32> : tensor<128xi32>
  %rem = arith.remsi %range, %c32 : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}
// CHECK-LABEL: @no_convert_negative_range
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 6: Non-constant divisor -> do NOT convert
module {
tt.func public @no_convert_non_constant_divisor(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %rem = arith.remsi %pid, %arg0 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_non_constant_divisor
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 7: divsi with unknown dividend -> do NOT convert
module {
tt.func public @no_convert_divsi_unknown_arg(%arg0: i32) -> i32 {
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %arg0, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_divsi_unknown_arg
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 8: divsi with negative divisor -> do NOT convert
module {
tt.func public @no_convert_divsi_negative_divisor() -> i32 {
  %pid = tt.get_program_id x : i32
  %cn128 = arith.constant -128 : i32
  %div = arith.divsi %pid, %cn128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_divsi_negative_divisor
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 9: divsi with non-constant divisor -> do NOT convert
module {
tt.func public @no_convert_divsi_non_constant_divisor(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %div = arith.divsi %pid, %arg0 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_divsi_non_constant_divisor
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 10: remsi with negative dividend -> result NOT non-negative
// Downstream divsi should NOT be converted.
module {
tt.func public @no_convert_remsi_neg_dividend(%arg0: i32) -> i32 {
  %rem = arith.remsi %arg0, %arg0 : i32
  %c32 = arith.constant 32 : i32
  %div = arith.divsi %rem, %c32 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_remsi_neg_dividend
// CHECK: arith.remsi
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 11: divsi with non-negative dividend but unknown divisor
// -> result NOT non-negative. Downstream remsi should NOT be converted.
module {
tt.func public @no_convert_divsi_unknown_divisor_chain(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %div = arith.divsi %pid, %arg0 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %div, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_divsi_unknown_divisor_chain
// CHECK: arith.divsi
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 12: maxsi with both operands unknown -> NOT non-negative.
// Downstream divsi should NOT be converted.
module {
tt.func public @no_convert_maxsi_both_unknown(%arg0: i32, %arg1: i32) -> i32 {
  %max = arith.maxsi %arg0, %arg1 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %max, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_maxsi_both_unknown
// CHECK: arith.maxsi
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 13: andi with both operands unknown -> do NOT convert
module {
tt.func public @no_convert_andi_both_unknown(%arg0: i32, %arg1: i32) -> i32 {
  %m = arith.andi %arg0, %arg1 : i32
  %c128 = arith.constant 128 : i32
  %rem = arith.remsi %m, %c128 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_andi_both_unknown
// CHECK: arith.andi
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 14: extsi with unknown input -> do NOT convert
module {
tt.func public @no_convert_extsi_unknown_input(%arg0: i16) -> i32 {
  %e = arith.extsi %arg0 : i16 to i32
  %c8 = arith.constant 8 : i32
  %div = arith.divsi %e, %c8 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_extsi_unknown_input
// CHECK: arith.extsi
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 15: shrui does not guarantee non-negativity -> do NOT convert
module {
tt.func public @no_convert_shrui_shift_zero(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %s = arith.shrui %arg0, %c0 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %s, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_shrui_shift_zero
// CHECK: arith.shrui
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 16: shrsi with unknown LHS -> do NOT convert
module {
tt.func public @no_convert_shrsi_unknown_lhs(%arg0: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %s = arith.shrsi %arg0, %c2 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %s, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_shrsi_unknown_lhs
// CHECK: arith.shrsi
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 17: trunci is unsafe (can truncate sign bit) -> do NOT convert
module {
tt.func public @no_convert_trunci() -> i16 {
  %pid = tt.get_program_id x : i32
  %t = arith.trunci %pid : i32 to i16
  %c8 = arith.constant 8 : i16
  %rem = arith.remsi %t, %c8 : i16
  tt.return %rem : i16
}
// CHECK-LABEL: @no_convert_trunci
// CHECK: arith.trunci
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 18: minui does not guarantee non-negativity -> do NOT convert
module {
tt.func public @no_convert_minui(%arg0: i32, %arg1: i32) -> i32 {
  %m = arith.minui %arg0, %arg1 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %m, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_minui
// CHECK: arith.minui
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 19: maxui does not guarantee non-negativity -> do NOT convert
module {
tt.func public @no_convert_maxui(%arg0: i32, %arg1: i32) -> i32 {
  %m = arith.maxui %arg0, %arg1 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %m, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_maxui
// CHECK: arith.maxui
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 20: minsi with one unknown operand -> do NOT convert
module {
tt.func public @no_convert_minsi_one_unknown(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %m = arith.minsi %pid, %arg0 : i32
  %c128 = arith.constant 128 : i32
  %div = arith.divsi %m, %c128 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_minsi_one_unknown
// CHECK: arith.minsi
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 21: select with false operand unknown -> do NOT convert
module {
tt.func public @no_convert_select_false_unknown(%cond: i1, %arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %s = arith.select %cond, %pid, %arg0 : i32
  %c64 = arith.constant 64 : i32
  %rem = arith.remsi %s, %c64 : i32
  tt.return %rem : i32
}
// CHECK-LABEL: @no_convert_select_false_unknown
// CHECK: arith.select
// CHECK: arith.remsi
// CHECK-NOT: arith.remui
}

// -----

// Negative Test 22: divsi by program_id (can be 0) -> do NOT convert
module {
tt.func public @no_convert_divsi_by_program_id() -> i32 {
  %pid0 = tt.get_program_id x : i32
  %pid1 = tt.get_program_id y : i32
  %div = arith.divsi %pid0, %pid1 : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_divsi_by_program_id
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 23: divsi by maxsi with zero constant (non-neg but not strictly positive) -> do NOT convert
module {
tt.func public @no_convert_divsi_by_maxsi_zero_const(%arg0: i32) -> i32 {
  %pid = tt.get_program_id x : i32
  %c0 = arith.constant 0 : i32
  %mx = arith.maxsi %arg0, %c0 : i32
  %div = arith.divsi %pid, %mx : i32
  tt.return %div : i32
}
// CHECK-LABEL: @no_convert_divsi_by_maxsi_zero_const
// CHECK: arith.maxsi
// CHECK: arith.divsi
// CHECK-NOT: arith.divui
}

// -----

// Negative Test 24: ceildivsi with unknown dividend -> do NOT convert
module {
tt.func public @no_convert_ceildivsi_unknown_dividend(%arg0: i32) -> i32 {
  %c128 = arith.constant 128 : i32
  %cdiv = arith.ceildivsi %arg0, %c128 : i32
  tt.return %cdiv : i32
}
// CHECK-LABEL: @no_convert_ceildivsi_unknown_dividend
// CHECK: arith.ceildivsi
// CHECK-NOT: arith.ceildivui
}

// -----

// Negative Test 25: ceildivsi with negative divisor -> do NOT convert
module {
tt.func public @no_convert_ceildivsi_negative_divisor() -> i32 {
  %pid = tt.get_program_id x : i32
  %cn128 = arith.constant -128 : i32
  %cdiv = arith.ceildivsi %pid, %cn128 : i32
  tt.return %cdiv : i32
}
// CHECK-LABEL: @no_convert_ceildivsi_negative_divisor
// CHECK: arith.ceildivsi
// CHECK-NOT: arith.ceildivui
}
