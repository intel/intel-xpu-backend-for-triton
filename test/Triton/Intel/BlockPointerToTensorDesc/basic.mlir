// RUN: triton-opt %s -triton-intel-block-pointer-to-tdesc | FileCheck %s

// CHECK-NOT: tt.make_tensor_ptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.advance
tt.func public @load(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @load
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32)
  // CHECK-DAG:     [[shape0:%.*]] = arith.trunci [[shape0_i64]] : i64 to i32
  // CHECK-DAG:     [[shape1:%.*]] = arith.trunci [[shape1_i64]] : i64 to i32
  // CHECK:         [[tdesc:%.*]] = tt.make_tensor_descriptor [[ptr]], [[[shape0]], [[shape1]]], [[[stride0]], [[stride1]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK:         tt.descriptor_load [[tdesc]][[[offset0]], [[offset1]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.load %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}

tt.func public @load_with_advances(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32, %offset2: i32, %offset3: i32, %offset4: i32, %offset5: i32) {
  // CHECK-LABEL: tt.func public @load_with_advances
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32, [[offset2:%.*]]: i32, [[offset3:%.*]]: i32, [[offset4:%.*]]: i32, [[offset5:%.*]]: i32)
  // CHECK-DAG:     [[shape0:%.*]] = arith.trunci [[shape0_i64]] : i64 to i32
  // CHECK-DAG:     [[shape1:%.*]] = arith.trunci [[shape1_i64]] : i64 to i32
  // CHECK:         [[tdesc:%.*]] = tt.make_tensor_descriptor [[ptr]], [[[shape0]], [[shape1]]], [[[stride0]], [[stride1]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK:         [[newoffset0:%.*]] = arith.addi [[offset0]], [[offset2]] : i32
  // CHECK:         [[newoffset1:%.*]] = arith.addi [[offset1]], [[offset3]] : i32
  // CHECK:         [[newoffset2:%.*]] = arith.addi [[newoffset0]], [[offset4]] : i32
  // CHECK:         [[newoffset3:%.*]] = arith.addi [[newoffset1]], [[offset5]] : i32
  // CHECK:         tt.descriptor_load [[tdesc]][[[newoffset2]], [[newoffset3]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1 = tt.advance %0, [%offset2, %offset3] : <tensor<256x32xbf16>>
  %2 = tt.advance %1, [%offset4, %offset5] : <tensor<256x32xbf16>>
  %3 = tt.load %2 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
  tt.return
}

tt.func public @load_in_for(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32, %offset2: i32, %offset3: i32) {
  // CHECK-LABEL: tt.func public @load_in_for
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32, [[offset2:%.*]]: i32, [[offset3:%.*]]: i32)
  // CHECK-DAG:     [[shape0:%.*]] = arith.trunci [[shape0_i64]] : i64 to i32
  // CHECK-DAG:     [[shape1:%.*]] = arith.trunci [[shape1_i64]] : i64 to i32
  // CHECK:         [[tdesc:%.*]] = tt.make_tensor_descriptor [[ptr]], [[[shape0]], [[shape1]]], [[[stride0]], [[stride1]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK:         scf.for {{.*}} iter_args([[newoffset0:%.*]] = [[offset0]], [[newoffset1:%.*]] = [[offset1]]) -> (i32, i32)  : i32 {
  // CHECK:           tt.descriptor_load [[tdesc]][[[newoffset0]], [[newoffset1]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
  // CHECK:           [[add0:%.*]] = arith.addi [[newoffset0]], [[offset2]] : i32
  // CHECK:           [[add1:%.*]] = arith.addi [[newoffset1]], [[offset3]] : i32
  // CHECK:           scf.yield [[add0]], [[add1]] : i32, i32
  // CHECK:         }
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c4096 = arith.constant 4096 : i32
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1:1 = scf.for %i = %c0 to %c4096 step %c32 iter_args(%ptr = %0) -> (!tt.ptr<tensor<256x32xbf16>>) : i32 {
    %2 = tt.load %ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
    %3 = tt.advance %ptr, [%offset2, %offset3] : <tensor<256x32xbf16>>
    scf.yield %3 : !tt.ptr<tensor<256x32xbf16>>
  }
  tt.return
}

tt.func public @for_result_used(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32, %offset2: i32, %offset3: i32) -> !tt.ptr<tensor<256x32xbf16>> {
  // CHECK-LABEL: tt.func public @for_result_used
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32, [[offset2:%.*]]: i32, [[offset3:%.*]]: i32)
  // CHECK:         [[ret:%.*]]:3 = scf.for {{.*}} iter_args([[ptr:%.*]] = {{.*}}, [[newoffset0:%.*]] = [[offset0]], [[newoffset1:%.*]] = [[offset1]]) -> (!tt.ptr<tensor<256x32xbf16>>, i32, i32)  : i32 {
  // CHECK:           [[advance:%.*]] = tt.advance [[ptr]], [[[offset2]], [[offset3]]] : <tensor<256x32xbf16>>
  // CHECK:           scf.yield [[advance]], {{.*}} : !tt.ptr<tensor<256x32xbf16>>, i32, i32
  // CHECK:         }
  // CHECK:         tt.return [[ret]]#0 : !tt.ptr<tensor<256x32xbf16>>
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c4096 = arith.constant 4096 : i32
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1:1 = scf.for %i = %c0 to %c4096 step %c32 iter_args(%ptr = %0) -> (!tt.ptr<tensor<256x32xbf16>>) : i32 {
    %2 = tt.load %ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
    %3 = tt.advance %ptr, [%offset2, %offset3] : <tensor<256x32xbf16>>
    scf.yield %3 : !tt.ptr<tensor<256x32xbf16>>
  }
  tt.return %1#0 : !tt.ptr<tensor<256x32xbf16>>
}

tt.func public @no_advance_in_for(%arg0: !tt.ptr<bf16>, %shape0: i64, %shape1: i64, %stride0: i64, %stride1: i64, %offset0: i32, %offset1: i32) {
  // CHECK-LABEL: tt.func public @no_advance_in_for
  // CHECK-SAME: ([[ptr:%.*]]: !tt.ptr<bf16>, [[shape0_i64:%.*]]: i64, [[shape1_i64:%.*]]: i64, [[stride0:%.*]]: i64, [[stride1:%.*]]: i64, [[offset0:%.*]]: i32, [[offset1:%.*]]: i32)
  // CHECK-DAG:     [[shape0:%.*]] = arith.trunci [[shape0_i64]] : i64 to i32
  // CHECK-DAG:     [[shape1:%.*]] = arith.trunci [[shape1_i64]] : i64 to i32
  // CHECK:         [[tdesc:%.*]] = tt.make_tensor_descriptor [[ptr]], [[[shape0]], [[shape1]]], [[[stride0]], [[stride1]]] : <bf16>, <tensor<256x32xbf16>>
  // CHECK:         scf.for {{.*}} : i32 {
  // CHECK:           tt.descriptor_load [[tdesc]][[[offset0]], [[offset1]]] : !tt.tensordesc<tensor<256x32xbf16>> -> tensor<256x32xbf16>
  // CHECK:         }
  %c0 = arith.constant 0 : i32
  %c32 = arith.constant 32 : i32
  %c4096 = arith.constant 4096 : i32
  %0 = tt.make_tensor_ptr %arg0, [%shape0, %shape1], [%stride0, %stride1], [%offset0, %offset1] {order = array<i32: 1, 0>} : <tensor<256x32xbf16>>
  %1:1 = scf.for %i = %c0 to %c4096 step %c32 iter_args(%ptr = %0) -> (!tt.ptr<tensor<256x32xbf16>>) : i32 {
    %2 = tt.load %ptr {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xbf16>>
    scf.yield %ptr : !tt.ptr<tensor<256x32xbf16>>
  }
  tt.return
}
