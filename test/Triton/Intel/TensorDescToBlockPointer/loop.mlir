// RUN: triton-opt %s -triton-intel-tdesc-to-block-pointer  | FileCheck %s

module {
  // COM: Loop containing a tensor descriptor load operation using a loop invariant tensor descriptor.
  tt.func public @load_in_loop1(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out, %sum_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc, %sum_iter = %cst) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {
      %cast_i = arith.index_cast %i : index to i32
      %load1 = tt.descriptor_load %ptr_iter[%c8_i32, %cast_i] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16>
      %sum_next = arith.addf %sum_iter, %load1 : tensor<16x32xf16>
      scf.yield %ptr_iter, %sum_next : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    }
    tt.return
  }
  // CHECK:      tt.func public @load_in_loop1([[PARAM_0:%.+]]: !tt.ptr<f16>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x32xf16>>
  // CHECK:        [[FOR_RES:%.+]]:2 = scf.for [[IV:%.+]] = {{.*}} iter_args([[VAR_arg1:%.+]] = [[TENSOR_PTR]], [[VAR_arg2:%.+]] = [[CST]]) -> (!tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>) {
  // CHECK:          [[IDX_CAST:%.+]] = arith.index_cast [[IV]] : index to i32
  // CHECK:          [[TENSOR_PTR_1:%.+]] = tt.advance [[VAR_arg1]], {{\[}}[[CST_8_i32]], [[IDX_CAST]]] : <tensor<16x32xf16>>
  // CHECK:          [[LOAD:%.+]] = tt.load [[TENSOR_PTR_1]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:          [[ADD:%.+]] = arith.addf [[VAR_arg2]], [[LOAD]] : tensor<16x32xf16>
  // CHECK:          scf.yield [[VAR_arg1]], [[ADD]] : !tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>
  // CHECK:        }
  // CHECK:        tt.return
  // CHECK:      }

  // COM: Loop containing a tensor descriptor load operation using a loop variant tensor descriptor.
  tt.func public @load_in_loop2(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out, %sum_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc, %sum_iter = %cst) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {
      %cast_i = arith.index_cast %i : index to i32
      %load1 = tt.descriptor_load %ptr_iter[%c8_i32, %cast_i] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16>
      %sum_next = arith.addf %sum_iter, %load1 : tensor<16x32xf16>
      %tdesc_in_loop = tt.make_tensor_descriptor %arg0, [%arg2, %arg1], [%c1_i64, %0] : <f16>, <tensor<16x32xf16>>
      %cmp = arith.cmpi eq, %cast_i, %c8_i32 : i32
      %sel_tdesc = arith.select %cmp, %ptr_iter, %tdesc_in_loop : !tt.tensordesc<tensor<16x32xf16>>
      scf.yield %sel_tdesc, %sum_next : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    }
    tt.return
  }
  // CHECK:      tt.func public @load_in_loop2({{.*}}) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x32xf16>>
  // CHECK:        [[FOR_RES:%.+]]:2 = scf.for [[IV:%.+]] = {{.*}} iter_args([[VAR_arg1:%.+]] = [[TENSOR_PTR]], [[VAR_arg2:%.+]] = [[CST]]) -> (!tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>) {
  // CHECK:          [[IDX_CAST:%.+]] = arith.index_cast [[IV]] : index to i32
  // CHECK:          [[TENSOR_PTR_1:%.+]] = tt.advance [[VAR_arg1]], {{\[}}[[CST_8_i32]], [[IDX_CAST]]] : <tensor<16x32xf16>>
  // CHECK:          [[LOAD:%.+]] = tt.load [[TENSOR_PTR_1]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:          [[ADD:%.+]] = arith.addf [[VAR_arg2]], [[LOAD]] : tensor<16x32xf16>
  // CHECK-DAG:      [[EXTSI_PARAM_1a:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:      [[EXTSI_PARAM_2a:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK-DAG:      [[CST_0_i32_1:%.+]] = arith.constant 0 : i32
  // CHECK:          [[TENSOR_PTR2:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_2a]], [[EXTSI_PARAM_1a]]], {{\[}}[[CST_1_i64]], [[EXTSI_PARAM_2]]], {{\[}}[[CST_0_i32_1]], [[CST_0_i32_1]]] {{.*}} : <tensor<16x32xf16>>
  // CHECK:          [[CMP:%.+]] = arith.cmpi eq, [[IDX_CAST]], [[CST_8_i32]] : i32
  // CHECK:          [[TENSOR_PTR3:%.+]] = arith.select [[CMP]], [[VAR_arg1]], [[TENSOR_PTR:%.+]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:          scf.yield [[TENSOR_PTR3]], [[ADD]] : !tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>
  // CHECK:        }
  // CHECK:        tt.return
  // CHECK:      }

  // COM: Loop yields a tensor descriptor used by a tensor descriptor load.
  tt.func public @load_uses_loop_result(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc) -> (!tt.tensordesc<tensor<16x32xf16>>) {
      scf.yield %ptr_iter : !tt.tensordesc<tensor<16x32xf16>>
    }
    %cast_c10 = arith.index_cast %c10 : index to i32
    %load2 = tt.descriptor_load %tdesc_out[%c8_i32, %cast_c10] : !tt.tensordesc<tensor<16x32xf16>> -> tensor<16x32xf16>
    tt.return
  }
  // CHECK:      tt.func public @load_uses_loop_result({{.*}}) {
  // CHECK-NOT:    tt.load
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr {{.*}} : <tensor<16x32xf16>>
  // CHECK:        [[FOR_RES:%.+]] = scf.for [[IV:%.+]] = {{.*}} iter_args([[VAR_arg1:%.+]] = [[TENSOR_PTR]]) -> (!tt.ptr<tensor<16x32xf16>>)
  // CHECK:        [[TENSOR_PTR1:%.+]] = tt.advance [[FOR_RES]], {{.*}} : <tensor<16x32xf16>>
  // CHECK:        tt.load [[TENSOR_PTR1]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:        tt.return
  // CHECK:      }

  // COM: Loop containing a tensor descriptor store operation using a loop invariant tensor descriptor.
  tt.func public @store_in_loop1(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out, %sum_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc, %sum_iter = %cst) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {
      %cast_i = arith.index_cast %i : index to i32
      tt.descriptor_store %ptr_iter[%c8_i32, %cast_i], %sum_iter : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
      %sum_next = arith.addf %sum_iter, %cst : tensor<16x32xf16>
      scf.yield %ptr_iter, %sum_next : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    }
    tt.return
  }
  // CHECK:      tt.func public @store_in_loop1([[PARAM_0:%.+]]: !tt.ptr<f16>, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK-DAG:    [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
  // CHECK-DAG:    [[CST_8_i32:%.+]] = arith.constant 8 : i32
  // CHECK-DAG:    [[CST:%.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
  // CHECK-DAG:    [[EXTSI_PARAM_1:%.+]] = arith.extsi [[PARAM_1]] : i32 to i64
  // CHECK-DAG:    [[EXTSI_PARAM_2:%.+]] = arith.extsi [[PARAM_2]] : i32 to i64
  // CHECK:        [[TENSOR_PTR:%.+]] = tt.make_tensor_ptr [[PARAM_0]], {{\[}}[[EXTSI_PARAM_1]], [[EXTSI_PARAM_2]]], {{\[}}[[EXTSI_PARAM_2]], [[CST_1_i64]]], {{\[}}[[CST_0_i32]], [[CST_0_i32]]] {{.*}} : <tensor<16x32xf16>>
  // CHECK:        [[FOR_RES:%.+]]:2 = scf.for [[IV:%.+]] = {{.*}} iter_args([[VAR_arg1:%.+]] = [[TENSOR_PTR]], [[VAR_arg2:%.+]] = [[CST]]) -> (!tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>) {
  // CHECK:          [[IDX_CAST_1:%.+]] = arith.index_cast [[IV]] : index to i32
  // CHECK:          [[TENSOR_PTR_1:%.+]] = tt.advance [[VAR_arg1]], {{\[}}[[CST_8_i32]], [[IDX_CAST]]] : <tensor<16x32xf16>>
  // CHECK:          tt.store [[TENSOR_PTR_1]], [[VAR_arg2]] : !tt.ptr<tensor<16x32xf16>>
  // CHECK:          [[ADD:%.+]] = arith.addf [[VAR_arg2]], [[CST]] : tensor<16x32xf16>
  // CHECK:          scf.yield [[VAR_arg1]], [[ADD]] : !tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>
  // CHECK:        }
  // CHECK:        tt.return
  // CHECK:      }

  // COM: Loop containing a tensor descriptor store operation using a loop variant tensor descriptor.
  tt.func public @store_in_loop2(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out, %sum_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc, %sum_iter = %cst) -> (!tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>) {
      %cast_i = arith.index_cast %i : index to i32
      tt.descriptor_store %ptr_iter[%c8_i32, %cast_i], %sum_iter : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
      %sum_next = arith.addf %sum_iter, %cst : tensor<16x32xf16>
      %tdesc_in_loop = tt.make_tensor_descriptor %arg0, [%arg2, %arg1], [%c1_i64, %0] : <f16>, <tensor<16x32xf16>>
      %cmp = arith.cmpi eq, %cast_i, %c8_i32 : i32
      %sel_tdesc = arith.select %cmp, %ptr_iter, %tdesc_in_loop : !tt.tensordesc<tensor<16x32xf16>>
      scf.yield %sel_tdesc, %sum_next : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    }
    tt.return
  }
  // CHECK:      tt.func public @store_in_loop2({{.*}}) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK:        tt.make_tensor_ptr
  // CHECK:        [[FOR_RES:%.+]]:2 = scf.for [[IV:%.+]] = {{.*}} -> (!tt.ptr<tensor<16x32xf16>>, tensor<16x32xf16>) {
  // CHECK:          tt.advance
  // CHECK:          tt.store
  // CHECK:        }
  // CHECK:        tt.return
  // CHECK:      }

  // COM: Loop yields a tensor descriptor used by a tensor descriptor store.
  tt.func public @store_uses_loop_result(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf16>
    %0 = arith.extsi %arg2 : i32 to i64
    %tdesc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <f16>, <tensor<16x32xf16>>
    %tdesc_out = scf.for %i = %c0 to %c10 step %c1 iter_args(%ptr_iter = %tdesc) -> (!tt.tensordesc<tensor<16x32xf16>>) {
      scf.yield %ptr_iter : !tt.tensordesc<tensor<16x32xf16>>
    }
    %cast_c10 = arith.index_cast %c10 : index to i32
    tt.descriptor_store %tdesc_out[%c8_i32, %cast_c10], %cst : !tt.tensordesc<tensor<16x32xf16>>, tensor<16x32xf16>
    tt.return
  }
  // CHECK:      tt.func public @store_uses_loop_result({{.*}}) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_store
  // CHECK:        tt.make_tensor_ptr
  // CHECK:        tt.advance
  // CHECK:        tt.store
  // CHECK:        tt.return
  // CHECK:      }

  // COM: While loop contains a descriptor load operation.
  tt.func public @load_in_while_loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.get_program_id x : i32
    %3 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%c1_i64, %c1_i64] : <f32>, <tensor<8x128xf32>>
    %5 = scf.while (%arg3 = %3) : (!tt.tensordesc<tensor<8x128xf32>>) -> (!tt.tensordesc<tensor<8x128xf32>>) {
      %6 = arith.cmpi slt, %c0_i32, %arg2 : i32
      scf.condition(%6) %arg3 : !tt.tensordesc<tensor<8x128xf32>>
    } do {
    ^bb0(%arg3: !tt.tensordesc<tensor<8x128xf32>>):
      %12 = tt.descriptor_load %arg3[%0, %c0_i32] : !tt.tensordesc<tensor<8x128xf32>> -> tensor<8x128xf32>
      scf.yield %arg3 : !tt.tensordesc<tensor<8x128xf32>>
    }
    tt.return
  }
  // CHECK: tt.func public @load_in_while_loop({{.*}}) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK:        [[TENSOR_PTR:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<8x128xf32>
  // CHECK:        scf.while ([[ARG3:%.*]] = [[TENSOR_PTR]]) : (!tt.ptr<tensor<8x128xf32>>) -> !tt.ptr<tensor<8x128xf32>> {
  // CHECK:          scf.condition({{.*}}) [[ARG3]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:        } do {
  // CHECK:        ^bb0([[ARG4:%.*]]: !tt.ptr<tensor<8x128xf32>>):
  // CHECK:          [[PTR1:%.*]] = tt.advance [[ARG4]], {{.*}} : <tensor<8x128xf32>
  // CHECK:          tt.load [[PTR1]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:          scf.yield [[ARG4]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:        }

  // COM: For loop yields a tensor descriptor used by a while loop.
  tt.func public @while_uses_tdesc_yielded_by_for_loop(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %2 = arith.extsi %arg2 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%2, %c1_i64] : <f32>, <tensor<8x128xf32>>
    %4 = scf.for %arg3 = %c0_i32 to %arg2 step %c128_i32 iter_args(%arg4 = %3) -> (!tt.tensordesc<tensor<8x128xf32>>) : i32 {
      scf.yield %arg4 : !tt.tensordesc<tensor<8x128xf32>>
    }
    %5 = scf.while (%arg3 = %4) : (!tt.tensordesc<tensor<8x128xf32>>) -> (!tt.tensordesc<tensor<8x128xf32>>) {
      %6 = arith.cmpi slt, %0, %arg2 : i32
      scf.condition(%6) %arg3 : !tt.tensordesc<tensor<8x128xf32>>
    } do {
    ^bb0(%arg3: !tt.tensordesc<tensor<8x128xf32>>):
      %12 = tt.descriptor_load %arg3[%c8_i32, %c8_i32] : !tt.tensordesc<tensor<8x128xf32>> -> tensor<8x128xf32>
      scf.yield %arg3 : !tt.tensordesc<tensor<8x128xf32>>
    }
    tt.return
  }
  // CHECK:      tt.func public @while_uses_tdesc_yielded_by_for_loop({{.*}}) {
  // CHECK-NOT:    tt.make_tensor_descriptor
  // CHECK-NOT:    tt.descriptor_load
  // CHECK:        [[TENSOR_PTR:%.*]] = tt.make_tensor_ptr {{.*}} : <tensor<8x128xf32>
  // CHECK:        [[FOR_RES:%.+]] = scf.for [[IV:%.+]] = {{.*}} iter_args([[ARG3:%.*]] = [[TENSOR_PTR]]) -> (!tt.ptr<tensor<8x128xf32>>) : i32 {
  // CHECK:          scf.yield {{.*}} : !tt.ptr<tensor<8x128xf32>>
  // CHECK:        }
  // CHECK:        scf.while ([[ARG3:%.*]] = [[FOR_RES]]) : (!tt.ptr<tensor<8x128xf32>>) -> !tt.ptr<tensor<8x128xf32>> {
  // CHECK:          scf.condition({{.*}}) [[ARG3]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:        } do {
  // CHECK:        ^bb0([[ARG4:%.*]]: !tt.ptr<tensor<8x128xf32>>):
  // CHECK:          [[PTR1:%.*]] = tt.advance [[ARG4]], {{.*}} : <tensor<8x128xf32>
  // CHECK:          tt.load [[PTR1]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:          scf.yield [[ARG4]] : !tt.ptr<tensor<8x128xf32>>
  // CHECK:        }

}
