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
  // CHECK-DAG:       [[VAR_8:%.+]] = arith.cmpi sgt, [[PARAM_3]], [[CST_1023]] : i32
  // CHECK-DAG:       [[CST_511:%.+]] = arith.constant 511 : i32
  // CHECK-DAG:       [[VAR_9:%.+]] = arith.cmpi sgt, [[PARAM_3]], [[CST_511]] : i32
  // CHECK:           [[VAR_10:%.+]] = arith.andi [[VAR_8]], [[VAR_9]] : i1 | [[VAR_10:%.+]] = arith.andi [[VAR_9]], [[VAR_8]] : i1
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
