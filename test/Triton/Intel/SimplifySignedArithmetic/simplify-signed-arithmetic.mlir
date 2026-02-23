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
