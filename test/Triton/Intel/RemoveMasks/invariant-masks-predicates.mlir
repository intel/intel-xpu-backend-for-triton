// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  // COM: Test loop versioning for loads with invariant mask using sle predicate
  // COM: Mask in form [0..END] <= splat(X)
  tt.func public @test_invariant_masks_sle(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi sle, %2, %3 : tensor<1024xi32>
    scf.for %arg4 = %0 to %arg2 step %1  : i32 {
      %5 = arith.muli %arg4, %arg1 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %4, %cst : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_sle([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi sle, [[VAR_2]], [[VAR_3]] : tensor<1024xi32>
  // CHECK-DAG:       [[CST_1023:%.+]] = arith.constant 1023 : i32
  // CHECK:           [[VAR_5:%.+]] = arith.cmpi sge, [[PARAM_3]], [[CST_1023]] : i32
  // CHECK:           scf.if [[VAR_5]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK:               [[LOAD:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: Test loop versioning for loads with invariant mask using ult predicate
  // COM: Mask in form [0..END] <u splat(X)
  tt.func public @test_invariant_masks_ult(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi ult, %2, %3 : tensor<1024xi32>
    scf.for %arg4 = %0 to %arg2 step %1  : i32 {
      %5 = arith.muli %arg4, %arg1 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %4, %cst : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_ult([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi ult, [[VAR_2]], [[VAR_3]] : tensor<1024xi32>
  // CHECK-DAG:       [[CST_1023:%.+]] = arith.constant 1023 : i32
  // CHECK:           [[VAR_5:%.+]] = arith.cmpi ugt, [[PARAM_3]], [[CST_1023]] : i32
  // CHECK:           scf.if [[VAR_5]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK:               [[LOAD:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: Test loop versioning for loads with invariant mask using ule predicate
  // COM: Mask in form [0..END] <=u splat(X)
  tt.func public @test_invariant_masks_ule(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi ule, %2, %3 : tensor<1024xi32>
    scf.for %arg4 = %0 to %arg2 step %1  : i32 {
      %5 = arith.muli %arg4, %arg1 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %4, %cst : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_ule([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi ule, [[VAR_2]], [[VAR_3]] : tensor<1024xi32>
  // CHECK-DAG:       [[CST_1023:%.+]] = arith.constant 1023 : i32
  // CHECK:           [[VAR_5:%.+]] = arith.cmpi uge, [[PARAM_3]], [[CST_1023]] : i32
  // CHECK:           scf.if [[VAR_5]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK:               [[LOAD:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: Test reversed form: splat(X) <= [0..END] with sle predicate
  tt.func public @test_invariant_masks_sle_reversed(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = arith.cmpi sle, %3, %2 : tensor<1024xi32>
    scf.for %arg4 = %0 to %arg2 step %1  : i32 {
      %5 = arith.muli %arg4, %arg1 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %8 = tt.addptr %7, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %9 = tt.load %8, %4, %cst : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_sle_reversed([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<1024xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi sle, [[VAR_3]], [[VAR_2]] : tensor<1024xi32>
  // CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : i32
  // CHECK:           [[VAR_5:%.+]] = arith.cmpi sle, [[PARAM_3]], [[CST_0]] : i32
  // CHECK:           scf.if [[VAR_5]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK:               [[LOAD:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<1024x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: Test reversed form: splat(X) <u [0..END] with ult predicate
  tt.func public @test_invariant_masks_ult_reversed(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<512xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %4 = arith.cmpi ult, %3, %2 : tensor<512xi32>
    scf.for %arg4 = %0 to %arg2 step %1  : i32 {
      %5 = arith.muli %arg4, %arg1 : i32
      %6 = tt.addptr %arg0, %5 : !tt.ptr<f32>, i32
      %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
      %8 = tt.addptr %7, %2 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
      %9 = tt.load %8, %4, %cst : tensor<512x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_invariant_masks_ult_reversed([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<512xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi ult, [[VAR_3]], [[VAR_2]] : tensor<512xi32>
  // CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : i32
  // CHECK:           [[VAR_5:%.+]] = arith.cmpi ult, [[PARAM_3]], [[CST_0]] : i32
  // CHECK:           scf.if [[VAR_5]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK:               [[LOAD:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<512x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}

// -----

module {
  // COM: Test with multiple invariant masks using different predicates in the same loop
  tt.func public @test_multiple_invariant_masks(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<0xFF800000> : tensor<512xf32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<256xf32>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %4 = arith.cmpi slt, %2, %3 : tensor<512xi32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<256xi32>
    %7 = arith.cmpi ule, %5, %6 : tensor<256xi32>
    scf.for %arg5 = %0 to %arg2 step %1  : i32 {
      %8 = arith.muli %arg5, %arg1 : i32
      %9 = tt.addptr %arg0, %8 : !tt.ptr<f32>, i32
      %10 = tt.splat %9 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
      %11 = tt.splat %9 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %12 = tt.addptr %10, %2 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
      %13 = tt.addptr %11, %5 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %14 = tt.load %12, %4, %cst : tensor<512x!tt.ptr<f32>>
      %15 = tt.load %13, %7, %cst_0 : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }

  // CHECK:         tt.func public @test_multiple_invariant_masks([[PARAM_0:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1:%.+]]: i32, [[PARAM_2:%.+]]: i32, [[PARAM_3:%.+]]: i32, [[PARAM_4:%.+]]: i32) {
  // CHECK:           [[VAR_2:%.+]] = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  // CHECK:           [[VAR_3:%.+]] = tt.splat [[PARAM_3]] : i32 -> tensor<512xi32>
  // CHECK:           [[VAR_4:%.+]] = arith.cmpi slt, [[VAR_2]], [[VAR_3]] : tensor<512xi32>
  // CHECK:           [[VAR_5:%.+]] = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  // CHECK:           [[VAR_6:%.+]] = tt.splat [[PARAM_4]] : i32 -> tensor<256xi32>
  // CHECK-DAG:       [[VAR_7:%.+]] = arith.cmpi ule, [[VAR_5]], [[VAR_6]] : tensor<256xi32>
  // CHECK-DAG:       [[CST_511:%.+]] = arith.constant 511 : i32
  // CHECK:           [[VAR_8:%.+]] = arith.cmpi sgt, [[PARAM_3]], [[CST_511]] : i32
  // CHECK:           [[CST_255:%.+]] = arith.constant 255 : i32
  // CHECK:           [[VAR_9:%.+]] = arith.cmpi uge, [[PARAM_4]], [[CST_255]] : i32
  // CHECK:           [[VAR_10:%.+]] = arith.andi [[VAR_8]], [[VAR_9]] : i1
  // CHECK:           scf.if [[VAR_10]] {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-NOT:           tt.load {{.*}}, {{.*}}, {{.*}}
  // CHECK:             }
  // CHECK:           } else {
  // CHECK:             scf.for {{.+}} = {{.+}} to {{.+}} step {{.+}} : i32 {
  // CHECK-DAG:           [[LOAD_A:%.+]] = tt.load {{.*}}, [[VAR_4]], {{.*}} : tensor<512x!tt.ptr<f32>>
  // CHECK-DAG:           [[LOAD_B:%.+]] = tt.load {{.*}}, [[VAR_7]], {{.*}} : tensor<256x!tt.ptr<f32>>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           tt.return
  // CHECK:         }
}
