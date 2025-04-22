// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  tt.func public @test_kernel1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x128xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<32> : tensor<64x32xi32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c64_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %16 = tt.splat %14 : i32 -> tensor<64xi32>
    %17 = arith.addi %16, %15 : tensor<64xi32>
    %18 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %19 = arith.remsi %17, %18 : tensor<64xi32>
    %20 = arith.muli %13, %c128_i32 : i32
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %22 = tt.splat %20 : i32 -> tensor<128xi32>
    %23 = arith.addi %22, %21 : tensor<128xi32>
    %24 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %25 = arith.remsi %23, %24 : tensor<128xi32>
    %26 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %27 = tt.expand_dims %19 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %28 = tt.splat %arg6 : i32 -> tensor<64x1xi32>
    %29 = arith.muli %27, %28 : tensor<64x1xi32>
    %30 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %31 = tt.broadcast %29 : tensor<64x1xi32> -> tensor<64x32xi32>
    %32 = tt.broadcast %30 : tensor<1x32xi32> -> tensor<64x32xi32>
    %33 = arith.addi %31, %32 : tensor<64x32xi32>
    %34 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>>
    %35 = tt.addptr %34, %33 : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
    %36 = tt.expand_dims %26 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %37 = tt.splat %arg7 : i32 -> tensor<32x1xi32>
    %38 = arith.muli %36, %37 : tensor<32x1xi32>
    %39 = tt.expand_dims %25 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %40 = tt.broadcast %38 : tensor<32x1xi32> -> tensor<32x128xi32>
    %41 = tt.broadcast %39 : tensor<1x128xi32> -> tensor<32x128xi32>
    %42 = arith.addi %40, %41 : tensor<32x128xi32>
    %43 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
    %44 = tt.addptr %43, %42 : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
    %45 = arith.addi %arg5, %c31_i32 : i32
    %46 = arith.divsi %45, %c32_i32 : i32
    %47 = arith.muli %arg7, %c32_i32 : i32
    %48 = tt.splat %47 : i32 -> tensor<32x128xi32>
    %49:3 = scf.for %arg9 = %c0_i32 to %46 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %35, %arg12 = %44) -> (tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>)  : i32 {
      %67 = arith.muli %arg9, %c32_i32 : i32
      %68 = arith.subi %arg5, %67 : i32
      %69 = tt.splat %68 : i32 -> tensor<1x32xi32>
      %70 = arith.cmpi slt, %30, %69 : tensor<1x32xi32>
      %71 = tt.broadcast %70 : tensor<1x32xi1> -> tensor<64x32xi1>
      %72 = tt.load %arg11, %71, %cst_1 : tensor<64x32x!tt.ptr<f16>>
      %73 = tt.splat %68 : i32 -> tensor<32x1xi32>
      %74 = arith.cmpi slt, %36, %73 : tensor<32x1xi32>
      %75 = tt.broadcast %74 : tensor<32x1xi1> -> tensor<32x128xi1>
      %76 = tt.load %arg12, %75, %cst_0 : tensor<32x128x!tt.ptr<f16>>
      %77 = tt.dot %72, %76, %arg10, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32>
      %78 = tt.addptr %arg11, %cst_2 : tensor<64x32x!tt.ptr<f16>>, tensor<64x32xi32>
      %79 = tt.addptr %arg12, %48 : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
      scf.yield %77, %78, %79 : tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>
    }
    %50 = arith.truncf %49#0 : tensor<64x128xf32> to tensor<64x128xf16>
    tt.return
  }

  // CHECK: tt.func public @test_kernel1([[PARAM_0_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32 {tt.divisibility = 16 : i32}, [[PARAM_5_:%.+]]: i32 {tt.divisibility = 16 : i32}, [[PARAM_6_:%.+]]: i32 {tt.divisibility = 16 : i32}, [[PARAM_7_:%.+]]: i32 {tt.divisibility = 16 : i32}, [[PARAM_8_:%.+]]: i32 {tt.divisibility = 16 : i32}) {
  // CHECK:   [[CST_0_i32:%.+]] = arith.constant 0 : i32
  // CHECK:   [[CST_32_i32:%.+]] = arith.constant 32 : i32
  // CHECK:   [[REM:%.+]] = arith.remsi [[PARAM_5_]], [[CST_32_i32]] : i32
  // CHECK:   [[CMP1:%.+]] = arith.cmpi eq, [[REM]], [[CST_0_i32]] : i32
  // CHECK:   [[CMP2:%.+]] = arith.cmpi sgt, [[PARAM_5_]], [[CST_32_i32]] : i32
  // CHECK:   [[VER_COND:%.+]] = arith.andi [[CMP1]], [[CMP2]] : i1
  // CHECK:   [[LOOP_VER:%.+]] = scf.if [[VER_COND]] -> (tensor<64x128xf32>) {
  // CHECK:     [[THEN_LOOP_RES:%.+]]:3 = scf.for {{.*}} iter_args([[VAR_arg10:%.+]] = {{.*}}, [[VAR_arg11:%.+]] = {{.*}}, [[VAR_arg12:%.+]] = {{.*}}) -> (tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>) : i32 {
  // CHECK:       [[LOAD_A1:%.+]] = tt.load [[VAR_arg11]] : tensor<64x32x!tt.ptr<f16>>
  // CHECK:       [[LOAD_B2:%.+]] = tt.load [[VAR_arg12]] : tensor<32x128x!tt.ptr<f16>>
  // CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>
  // CHECK:     }
  // CHECK:     scf.yield [[THEN_LOOP_RES]]#0 : tensor<64x128xf32>
  // CHECK:   } else {
  // CHECK:     [[ELSE_LOOP_RES:%.+]]:3 = scf.for {{.*}} iter_args([[VAR_arg10:%.+]] = {{.*}}, [[VAR_arg11:%.+]] = {{.*}}, [[VAR_arg12:%.+]] = {{.*}}) -> (tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>) : i32 {
  // CHECK:       [[LOAD_A2:%.+]] = tt.load [[VAR_arg11]], {{.*}}, {{.*}} : tensor<64x32x!tt.ptr<f16>>
  // CHECK:       [[LOAD_B2:%.+]] = tt.load [[VAR_arg12]], {{.*}}, {{.*}} : tensor<32x128x!tt.ptr<f16>>
  // CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<64x128xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>
  // CHECK:     }
  // CHECK:     scf.yield [[ELSE_LOOP_RES]]#0 : tensor<64x128xf32>
  // CHECK:   }
  // CHECK:   tt.return
  // CHECK: }

  tt.func public @test_kernel2(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x256xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<512> : tensor<1x256xi32>
    %cst_3 = arith.constant dense<32768> : tensor<64x256xi32>
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32>
    %c64_i32 = arith.constant 64 : i32
    %cst_5 = arith.constant dense<512> : tensor<64x1xi32>
    %cst_6 = arith.constant dense<512> : tensor<128x1xi32>
    %cst_7 = arith.constant dense<512> : tensor<256xi32>
    %c256_i32 = arith.constant 256 : i32
    %cst_8 = arith.constant dense<512> : tensor<128xi32>
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c8_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c4_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c8_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.addi %2, %6 : i32
    %8 = arith.divsi %5, %4 : i32
    %9 = arith.muli %7, %c128_i32 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %11 = tt.splat %9 : i32 -> tensor<128xi32>
    %12 = arith.addi %11, %10 : tensor<128xi32>
    %13 = arith.remsi %12, %cst_8 : tensor<128xi32>
    %14 = arith.muli %8, %c256_i32 : i32
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %16 = tt.splat %14 : i32 -> tensor<256xi32>
    %17 = arith.addi %16, %15 : tensor<256xi32>
    %18 = arith.remsi %17, %cst_7 : tensor<256xi32>
    %19 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %20 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %21 = arith.muli %20, %cst_6 : tensor<128x1xi32>
    %22 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %23 = tt.broadcast %21 : tensor<128x1xi32> -> tensor<128x64xi32>
    %24 = tt.broadcast %22 : tensor<1x64xi32> -> tensor<128x64xi32>
    %25 = arith.addi %23, %24 : tensor<128x64xi32>
    %26 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %27 = tt.addptr %26, %25 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
    %28 = tt.expand_dims %19 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %29 = arith.muli %28, %cst_5 : tensor<64x1xi32>
    %30 = tt.expand_dims %18 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %31 = tt.broadcast %29 : tensor<64x1xi32> -> tensor<64x256xi32>
    %32 = tt.broadcast %30 : tensor<1x256xi32> -> tensor<64x256xi32>
    %33 = arith.addi %31, %32 : tensor<64x256xi32>
    %34 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>>
    %35 = tt.addptr %34, %33 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
    %36:3 = scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg4 = %cst, %arg5 = %27, %arg6 = %35) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>)  : i32 {
      %51 = arith.muli %arg3, %c64_i32 : i32
      %52 = arith.subi %c512_i32, %51 : i32
      %53 = tt.splat %52 : i32 -> tensor<1x64xi32>
      %54 = arith.cmpi slt, %22, %53 : tensor<1x64xi32>
      %55 = tt.broadcast %54 : tensor<1x64xi1> -> tensor<128x64xi1>
      %56 = tt.load %arg5, %55, %cst_1 : tensor<128x64x!tt.ptr<f16>>
      %57 = tt.splat %52 : i32 -> tensor<64x1xi32>
      %58 = arith.cmpi slt, %28, %57 : tensor<64x1xi32>
      %59 = tt.broadcast %58 : tensor<64x1xi1> -> tensor<64x256xi1>
      %60 = tt.load %arg6, %59, %cst_0 : tensor<64x256x!tt.ptr<f16>>
      %61 = tt.dot %56, %60, %arg4, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
      %62 = tt.addptr %arg5, %cst_4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %63 = tt.addptr %arg6, %cst_3 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
      scf.yield %61, %62, %63 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
    }
    %37 = arith.truncf %36#0 : tensor<128x256xf32> to tensor<128x256xf16>
    tt.return
  }
  // CHECK:  tt.func public @test_kernel2([[PARAM_0_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  // CHECK:    [[VER_COND:%.+]] = arith.constant true
  // CHECK:    [[LOOP_VER:%.+]] = scf.if [[VER_COND]] -> (tensor<128x256xf32>) {
  // CHECK:      [[THEN_LOOP_RES:%.+]]:3 = scf.for {{.*}} iter_args([[VAR_arg4:%.+]] = {{.*}}, [[VAR_arg5:%.+]] = {{.*}}, [[VAR_arg6:%.+]] = {{.*}}) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>) : i32 {
  // CHECK:        [[LOAD_A1:%.+]] = tt.load [[VAR_arg5]] : tensor<128x64x!tt.ptr<f16>>
  // CHECK:        [[LOAD_B2:%.+]] = tt.load [[VAR_arg6]] : tensor<64x256x!tt.ptr<f16>>
  // CHECK:        scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
  // CHECK:      }
  // CHECK:      scf.yield [[THEN_LOOP_RES]]#0 : tensor<128x256xf32>
  // CHECK:    } else {
  // CHECK:      [[ELSE_LOOP_RES:%.+]]:3 = scf.for {{.*}} iter_args([[VAR_arg4:%.+]] = {{.*}}, [[VAR_arg5:%.+]] = {{.*}}, [[VAR_arg6:%.+]] = {{.*}}) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>) : i32 {
  // CHECK:        [[LOAD_A2:%.+]] = tt.load [[VAR_arg5]], {{.*}}, {{.*}} : tensor<128x64x!tt.ptr<f16>>
  // CHECK:        [[LOAD_B2:%.+]] = tt.load [[VAR_arg6]], {{.*}}, {{.*}} : tensor<64x256x!tt.ptr<f16>>
  // CHECK:        scf.yield {{.*}}, {{.*}}, {{.*}} : tensor<128x256xf32>, tensor<128x64x!tt.ptr<f16>>, tensor<64x256x!tt.ptr<f16>>
  // CHECK:      }
  // CHECK:      scf.yield [[ELSE_LOOP_RES]]#0 : tensor<128x256xf32>
  // CHECK:    }
  // CHECK:    tt.return

}
