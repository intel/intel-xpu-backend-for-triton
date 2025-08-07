// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: test loop versioning for loads with invariant mask operations.
  // COM: masks in form [0..END] < splat(X)
  tt.func public @test_invariant_masks_1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<512xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi slt, %2, %3 : tensor<1024xi32>
    %5 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %7 = arith.cmpi slt, %5, %6 : tensor<512xi32>
    scf.for %arg6 = %0 to %arg2 step %1  : i32 {
      %8 = arith.muli %arg6, %arg1 : i32
      %9 = tt.addptr %arg0, %8 : !tt.ptr<f32>, i32
      %10 = tt.splat %9 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %11 = tt.splat %9 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
      %12 = tt.addptr %10, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %13 = tt.addptr %11, %5 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
      %14 = tt.load %12, %4, %cst : tensor<1024x!tt.ptr<f32>>
      %15 = tt.load %13, %7, %cst_0 : tensor<512x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_1([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi slt, [[VAR_2]], [[VAR_3]] : tensor<1024xi32>
  // CHECK:           [[VAR_5:%.+]] = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  // CHECK:           [[VAR_6:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<512xi32>
  // CHECK-DAG:       [[VAR_7:%.+]] = arith.cmpi slt, [[VAR_5]], [[VAR_6]] : tensor<512xi32>
  // CHECK-DAG:       [[CST_1023:%.+]] = arith.constant 1023 : i32
  // CHECK:           [[VAR_8:%.+]] = arith.cmpi sgt, [[PARAM_3]], [[CST_1023]] : i32
  // CHECK:           [[CST_511:%.+]] = arith.constant 511 : i32
  // CHECK:           [[VAR_9:%.+]] = arith.cmpi sgt, [[PARAM_3]], [[CST_511]] : i32
  // CHECK:           [[VAR_10:%.+]] = arith.andi [[VAR_8]], [[VAR_9]] : i1
  // CHECK:           scf.if [[VAR_10]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-DAG:           [[LOAD_A2:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK-DAG:           [[LOAD_B2:%.+]] = tt.load {{.*}}, [[VAR_7]], {{.*}} : tensor<512x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: test loop versioning for loads with invariant mask operations.
  // COM: masks in form splat(X) < [0..END]
  tt.func public @test_invariant_masks_2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<512xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi slt, %3, %2 : tensor<1024xi32>
    %5 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %7 = arith.cmpi slt, %6, %5 : tensor<512xi32>
    scf.for %arg6 = %0 to %arg2 step %1  : i32 {
      %8 = arith.muli %arg6, %arg1 : i32
      %9 = tt.addptr %arg0, %8 : !tt.ptr<f32>, i32
      %10 = tt.splat %9 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %11 = tt.splat %9 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
      %12 = tt.addptr %10, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %13 = tt.addptr %11, %5 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
      %14 = tt.load %12, %4, %cst : tensor<1024x!tt.ptr<f32>>
      %15 = tt.load %13, %7, %cst_0 : tensor<512x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_2([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi slt, [[VAR_3]], [[VAR_2]] : tensor<1024xi32>
  // CHECK:           [[VAR_5:%.+]] = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  // CHECK:           [[VAR_6:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<512xi32>
  // CHECK-DAG:       [[VAR_7:%.+]] = arith.cmpi slt, [[VAR_6]], [[VAR_5]] : tensor<512xi32>
  // CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:       [[VAR_8:%.+]] = arith.cmpi slt, [[PARAM_3]], [[CST_0_i32]] : i32
  // CHECK-DAG:       [[CST_0_1_i32:%.+]] = arith.constant 0 : i32
  // CHECK-DAG:       [[VAR_9:%.+]] = arith.cmpi slt, [[PARAM_3]], [[CST_0_1_i32]] : i32
  // CHECK:           [[VAR_10:%.+]] = arith.andi [[VAR_8]], [[VAR_9]] : i1
  // CHECK:           scf.if [[VAR_10]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-DAG:           [[LOAD_A2:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK-DAG:           [[LOAD_B2:%.+]] = tt.load {{.*}}, [[VAR_7]], {{.*}} : tensor<512x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: From Liger-Kernel
  // COM: For details: https://github.com/intel/intel-xpu-backend-for-triton/issues/4796
  // CHECK-LABEL: _error_repro_kernel
  tt.func public @_error_repro_kernel(%input_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %input_row_stride: i32 {tt.divisibility = 16 : i32}, %temp_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %temp_row_stride: i32 {tt.divisibility = 16 : i32}, %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_rows: i32, %n_cols: i32 {tt.divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %col_offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %mask = tt.splat %n_cols : i32 -> tensor<64xi32>
    %mask_0 = arith.cmpi slt, %col_offsets, %mask : tensor<64xi32>
    // CHECK: %[[IF:.*]]:3 = scf.if %{{.*}} -> (!tt.ptr<f32>, !tt.ptr<f32>, tensor<64xf32>) {
    %output_row:3 = scf.for %_ = %c0_i32 to %n_rows step %c1_i32 iter_args(%input_ptr_1 = %input_ptr, %temp_ptr_2 = %temp_ptr, %output_row_3 = %cst) -> (!tt.ptr<f32>, !tt.ptr<f32>, tensor<64xf32>)  : i32 {
      %input_row = tt.splat %input_ptr_1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %input_row_4 = tt.addptr %input_row, %col_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %input_row_5 = tt.load %input_row_4, %mask_0, %cst : tensor<64x!tt.ptr<f32>>
      %temp_ptr_6 = tt.addptr %temp_ptr_2, %temp_row_stride : !tt.ptr<f32>, i32
      %output_row_7 = arith.addf %output_row_3, %input_row_5 : tensor<64xf32>
      %input_ptr_8 = tt.addptr %input_ptr_1, %input_row_stride : !tt.ptr<f32>, i32
      scf.yield %input_ptr_8, %temp_ptr_6, %output_row_7 : !tt.ptr<f32>, !tt.ptr<f32>, tensor<64xf32>
    }
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %1 = tt.addptr %0, %col_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    // CHECK: tt.store %{{.*}}, %[[IF]]#2, %{{.*}} : tensor<64x!tt.ptr<f32>>
    tt.store %1, %output_row#2, %mask_0 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
