// RUN: triton-opt %s -triton-raise-block-pointer -canonicalize | FileCheck %s

module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %4, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %2, %9 : i32
    %11 = arith.cmpi slt, %10, %c8_i32 : i32
    %12 = arith.select %11, %10, %c8_i32 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %19 = tt.splat %17 : i32 -> tensor<128xi32>
    %20 = arith.addi %19, %18 : tensor<128xi32>
    %21 = arith.muli %16, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %23 = tt.splat %21 : i32 -> tensor<256xi32>
    %24 = arith.addi %23, %22 : tensor<256xi32>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %26 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %30 = tt.splat %arg7 : i32 -> tensor<1x64xi32>
    %31 = arith.muli %29, %30 : tensor<1x64xi32>
    %32 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.broadcast %31 : tensor<1x64xi32> -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %35 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %37 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %41 = tt.splat %arg9 : i32 -> tensor<1x256xi32>
    %42 = arith.muli %40, %41 : tensor<1x256xi32>
    %43 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x256xi32>
    %44 = tt.broadcast %42 : tensor<1x256xi32> -> tensor<64x256xi32>
    %45 = arith.addi %43, %44 : tensor<64x256xi32>
    %46 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %45 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %48 = tt.splat %cst : f32 -> tensor<128x256xf32>
    %49 = arith.muli %arg7, %c64_i32 : i32
    %50 = tt.splat %49 : i32 -> tensor<128x64xi32>
    %51 = arith.muli %arg8, %c64_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<64x256xi32>
    %53:3 = scf.for %arg12 = %c0_i32 to %6 step %c1_i32 iter_args(%arg13 = %48, %arg14 = %36, %arg15 = %47) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>)  : i32 {
      %71 = tt.load %arg14 : tensor<128x64x!tt.ptr<bf16>>
      %72 = tt.load %arg15 : tensor<64x256x!tt.ptr<bf16>>
      %73 = tt.dot %71, %72, %48 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %74 = arith.addf %arg13, %73 : tensor<128x256xf32>
      %75 = tt.addptr %arg14, %50 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
      %76 = tt.addptr %arg15, %52 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
      scf.yield %74, %75, %76 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>
    }
    %54 = arith.truncf %53#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %55 = tt.splat %arg10 : i32 -> tensor<128x1xi32>
    %56 = arith.muli %55, %26 : tensor<128x1xi32>
    %57 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %59 = tt.splat %arg11 : i32 -> tensor<1x256xi32>
    %60 = arith.muli %59, %40 : tensor<1x256xi32>
    %61 = tt.broadcast %58 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x256x!tt.ptr<bf16>>
    %62 = tt.broadcast %60 : tensor<1x256xi32> -> tensor<128x256xi32>
    %63 = tt.addptr %61, %62 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %64 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %65 = arith.cmpi slt, %26, %64 : tensor<128x1xi32>
    %66 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %67 = arith.cmpi slt, %40, %66 : tensor<1x256xi32>
    %68 = tt.broadcast %65 : tensor<128x1xi1> -> tensor<128x256xi1>
    %69 = tt.broadcast %67 : tensor<1x256xi1> -> tensor<128x256xi1>
    %70 = arith.andi %68, %69 : tensor<128x256xi1>
    // TODO: add back once masked stores are supported
    // tt.store %63, %54, %70 : tensor<128x256x!tt.ptr<bf16>>
    tt.store %63, %54 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func public @matmul_kernel_0123456789101112131415([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
// CHECK-DAG:       [[CST_0_i64:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_256_i32:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_128_i32:%.+]] = arith.constant 128 : i32
// CHECK-DAG:       [[CST_64_i32:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_0_i32:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_i32:%.+]] = arith.constant 1 : i32
// CHECK:           [[VAR_17_:%.+]] = arith.muli {{.*}}, [[CST_128_i32]] : i32
// CHECK:           [[VAR_18_:%.+]] = arith.muli {{.*}}, [[CST_256_i32]] : i32
// CHECK:           [[VAR_20_:%.+]] = arith.extsi [[PARAM_6_]] : i32 to i64
// CHECK:           [[VAR_21_:%.+]] = arith.extsi [[PARAM_7_]] : i32 to i64
// CHECK:           [[VAR_22_:%.+]] = arith.divui {{.*}}, [[PARAM_6_]] : i32
// CHECK:           [[VAR_23_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_20_]], [[VAR_21_]]], {{\[}}[[VAR_22_]], [[CST_0_i32]]] {{.*}} : <tensor<128x64xbf16>>
// CHECK:           [[VAR_24_:%.+]] = arith.extsi [[PARAM_8_]] : i32 to i64
// CHECK:           [[VAR_25_:%.+]] = arith.muli {{.*}}, [[PARAM_9_]] : i32
// CHECK:           [[VAR_26_:%.+]] = arith.extsi [[PARAM_9_]] : i32 to i64
// CHECK:           [[VAR_27_:%.+]] = arith.divui [[VAR_25_]], [[PARAM_9_]] : i32
// CHECK:           [[VAR_28_:%.+]] = tt.make_tensor_ptr [[PARAM_1_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_24_]], [[VAR_26_]]], {{\[}}[[CST_0_i32]], [[VAR_27_]]] {{.*}} : <tensor<64x256xbf16>>
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.muli [[PARAM_7_]], [[CST_64_i32]] : i32
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.muli [[PARAM_8_]], [[CST_64_i32]] : i32
// CHECK:           [[VAR_31_:%.+]]:3 = scf.for {{.*}} iter_args([[VAR_arg13_:%.+]] = [[VAR_cst_]], [[VAR_arg14_:%.+]] = [[VAR_23_]], [[VAR_arg15_:%.+]] = [[VAR_28_]]) -> (tensor<128x256xf32>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<64x256xbf16>>)  : i32 {
// CHECK-DAG:         [[VAR_40_:%.+]] = tt.load [[VAR_arg14_]] : !tt.ptr<tensor<128x64xbf16>>
// CHECK-DAG:         [[VAR_41_:%.+]] = tt.load [[VAR_arg15_]] : !tt.ptr<tensor<64x256xbf16>>
// CHECK:             [[VAR_42_:%.+]] = tt.dot [[VAR_40_]], [[VAR_41_]], [[VAR_cst_]], inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.addf [[VAR_arg13_]], [[VAR_42_]] : tensor<128x256xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.divui [[VAR_29_]], [[PARAM_6_]] : i32
// CHECK-DAG:         [[VAR_45_:%.+]] = tt.advance [[VAR_arg14_]], {{\[}}[[VAR_44_]], [[CST_0_i32]]] : <tensor<128x64xbf16>>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.divui [[VAR_30_]], [[PARAM_8_]] : i32
// CHECK-DAG:         [[VAR_47_:%.+]] = tt.advance [[VAR_arg15_]], {{\[}}[[VAR_46_]], [[CST_0_i32]]] : <tensor<64x256xbf16>>
// CHECK:             scf.yield [[VAR_43_]], [[VAR_45_]], [[VAR_47_]] : tensor<128x256xf32>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<64x256xbf16>>
// CHECK:           }
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.truncf [[VAR_31_]]#0 : tensor<128x256xf32> to tensor<128x256xbf16>
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.muli [[VAR_17_]], [[PARAM_10_]] : i32
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.extsi [[PARAM_10_]] : i32 to i64
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.muli [[VAR_18_]], [[PARAM_11_]] : i32
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.extsi [[PARAM_11_]] : i32 to i64
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.divui [[VAR_33_]], [[PARAM_10_]] : i32
// CHECK-DAG:       [[VAR_38_:%.+]] = arith.divui [[VAR_35_]], [[PARAM_11_]] : i32
// CHECK:           [[VAR_39_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{\[}}[[CST_0_i64]], [[CST_0_i64]]], {{\[}}[[VAR_34_]], [[VAR_36_]]], {{\[}}[[VAR_37_]], [[VAR_38_]]] {{.*}} : <tensor<128x256xbf16>>
// CHECK:           tt.store [[VAR_39_]], [[VAR_32_]] : !tt.ptr<tensor<128x256xbf16>>
// CHECK:           tt.return
// CHECK:         }
