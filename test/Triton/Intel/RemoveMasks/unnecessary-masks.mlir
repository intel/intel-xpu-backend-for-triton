// RUN: triton-opt %s -triton-intel-remove-masks | FileCheck %s

module {
  tt.func public @test1(%in_ptr0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %in_ptr1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16>
    %c576_i32 = arith.constant 576 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant dense<9216> : tensor<32x1xi32>
    %cst_5 = arith.constant dense<16> : tensor<1x32xi32>
    %cst_6 = arith.constant dense<576> : tensor<1x32xi32>
    %cst_7 = arith.constant dense<0xFF800000> : tensor<32x32xf32>
    %cst_8 = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %4 = tt.splat %1 : i32 -> tensor<32x1xi32>
    %5 = arith.addi %4, %3 : tensor<32x1xi32>
    %6 = tt.expand_dims %2 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %7 = arith.remsi %5, %cst_8 : tensor<32x1xi32>
    %8 = arith.divsi %5, %cst_8 : tensor<32x1xi32>
    %9 = tt.splat %in_ptr1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>>
    %10 = tt.addptr %9, %7 : tensor<32x1x!tt.ptr<f16>>, tensor<32x1xi32>
    %11 = tt.load %10 evictionPolicy = evict_last : tensor<32x1x!tt.ptr<f16>>
    %12 = arith.extf %11 : tensor<32x1xf16> to tensor<32x1xf32>
    %13 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x32xi32>
    %14 = arith.muli %8, %cst_4 : tensor<32x1xi32>
    %15 = tt.broadcast %14 : tensor<32x1xi32> -> tensor<32x32xi32>
    %16 = tt.splat %in_ptr0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>>
    %17 = tt.broadcast %12 : tensor<32x1xf32> -> tensor<32x32xf32>
    %_tmp5 = scf.for %r0_offset = %c0_i32 to %c576_i32 step %c32_i32 iter_args(%_tmp5_9 = %cst_7) -> (tensor<32x32xf32>)  : i32 {
      %44 = tt.splat %r0_offset : i32 -> tensor<1x32xi32>
      %45 = arith.addi %44, %6 : tensor<1x32xi32>
      %46 = arith.cmpi slt, %45, %cst_6 : tensor<1x32xi32>
      %47 = arith.muli %45, %cst_5 : tensor<1x32xi32>
      %48 = tt.broadcast %47 : tensor<1x32xi32> -> tensor<32x32xi32>
      %49 = arith.addi %13, %48 : tensor<32x32xi32>
      %50 = arith.addi %49, %15 : tensor<32x32xi32>
      %51 = tt.addptr %16, %50 : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
      %52 = tt.broadcast %46 : tensor<1x32xi1> -> tensor<32x32xi1>
      %53 = tt.load %51, %52, %cst evictionPolicy = evict_last : tensor<32x32x!tt.ptr<f16>>
      %54 = arith.extf %53 : tensor<32x32xf16> to tensor<32x32xf32>
      %55 = arith.addf %54, %17 : tensor<32x32xf32>
      %mask = arith.cmpf ogt, %_tmp5_9, %55 : tensor<32x32xf32>
      %56 = arith.cmpf une, %_tmp5_9, %_tmp5_9 : tensor<32x32xf32>
      %mask_10 = arith.ori %mask, %56 : tensor<32x32xi1>
      %57 = arith.select %mask_10, %_tmp5_9, %55 : tensor<32x32xi1>, tensor<32x32xf32>
      %58 = arith.select %52, %57, %_tmp5_9 : tensor<32x32xi1>, tensor<32x32xf32>
      scf.yield %58 : tensor<32x32xf32>
    }
    tt.return
  }
  // CHECK: tt.func public @test1
  // CHECK:   scf.for
  // CHECK:     [[PTR:%.+]] = tt.addptr {{.*}} : tensor<32x32x!tt.ptr<f16>>, tensor<32x32xi32>
  // CHECK:     [[LOAD:%.+]] = tt.load [[PTR]] evictionPolicy = evict_last : tensor<32x32x!tt.ptr<f16>>
  // CHECK:     arith.extf [[LOAD]] : tensor<32x32xf16> to tensor<32x32xf32>
  // CHECK:     [[ORI:%.+]] = arith.ori {{.*}} : tensor<32x32xi1>
  // CHECK:     [[SEL:%.+]] = arith.select [[ORI]], {{.*}}, {{.*}} : tensor<32x32xi1>, tensor<32x32xf32>
  // CHECK:     scf.yield [[SEL]] : tensor<32x32xf32>

 tt.func public @test2(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x8xf32>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x8xf32>
    %cst_2 = arith.constant dense<16384> : tensor<64x1xi32>
    %cst_3 = arith.constant dense<128> : tensor<1x8xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant dense<128> : tensor<64x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %4 = tt.splat %1 : i32 -> tensor<64x1xi32>
    %5 = arith.addi %4, %3 : tensor<64x1xi32>
    %6 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %8 = arith.remsi %5, %cst_4 : tensor<64x1xi32>
    %9 = arith.divsi %5, %cst_4 : tensor<64x1xi32>
    %10 = tt.broadcast %8 : tensor<64x1xi32> -> tensor<64x8xi32>
    %11 = arith.muli %9, %cst_2 : tensor<64x1xi32>
    %12 = tt.broadcast %11 : tensor<64x1xi32> -> tensor<64x8xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    %14:3 = scf.for %arg6 = %c0_i32 to %c128_i32 step %c8_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst_1, %arg9 = %cst_1) -> (tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>)  : i32 {
      %25 = tt.splat %arg6 : i32 -> tensor<1x8xi32>
      %26 = arith.addi %25, %7 : tensor<1x8xi32>
      %27 = arith.cmpi slt, %26, %cst_3 : tensor<1x8xi32>
      %28 = arith.muli %26, %cst_3 : tensor<1x8xi32>
      %29 = tt.broadcast %28 : tensor<1x8xi32> -> tensor<64x8xi32>
      %30 = arith.addi %10, %29 : tensor<64x8xi32>
      %31 = arith.addi %30, %12 : tensor<64x8xi32>
      %32 = tt.addptr %13, %31 : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
      %33 = tt.broadcast %27 : tensor<1x8xi1> -> tensor<64x8xi1>
      %34 = tt.load %32, %33, %cst_1 evictionPolicy = evict_first : tensor<64x8x!tt.ptr<f32>>
      %35 = arith.cmpi eq, %arg6, %c0_i32 : i32
      %36:3 = scf.if %35 -> (tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>) {
        scf.yield %cst_1, %34, %cst_0 : tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>
      } else {
        %40 = arith.subf %34, %arg7 : tensor<64x8xf32>
        %41 = arith.addf %arg9, %cst_0 : tensor<64x8xf32>
        %42 = arith.divf %40, %41 : tensor<64x8xf32>
        %43 = arith.addf %arg7, %42 : tensor<64x8xf32>
        %44 = arith.subf %34, %43 : tensor<64x8xf32>
        %45 = arith.mulf %40, %44 : tensor<64x8xf32>
        %46 = arith.addf %arg8, %45 : tensor<64x8xf32>
        scf.yield %46, %43, %41 : tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>
      }
      %37 = arith.select %33, %36#1, %arg7 : tensor<64x8xi1>, tensor<64x8xf32>
      %38 = arith.select %33, %36#0, %arg8 : tensor<64x8xi1>, tensor<64x8xf32>
      %39 = arith.select %33, %36#2, %arg9 : tensor<64x8xi1>, tensor<64x8xf32>
      scf.yield %37, %38, %39 : tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>
    }
    tt.return
  }
  // CHECK: tt.func public @test2
  // CHECK:   scf.for
  // CHECK:     [[PTR:%.+]] = tt.addptr {{.*}} : tensor<64x8x!tt.ptr<f32>>, tensor<64x8xi32>
  // CHECK:     [[LOAD:%.+]] = tt.load [[PTR]] evictionPolicy = evict_first : tensor<64x8x!tt.ptr<f32>>
  // CHECK:     [[IF_RES:%.+]]:3 = scf.if {{.*}} -> (tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>)
  // CHECK:       scf.yield {{.*}}, [[LOAD]], {{.*}} : tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>
  // CHECK:     else
  // CHECK-2:     arith.subf [[LOAD]], {{.*}} : tensor<64x8xf32
  // CHECK:     }
  // CHECK:     scf.yield [[IF_RES]]#1, [[IF_RES]]#0, [[IF_RES]]#2 : tensor<64x8xf32>, tensor<64x8xf32>, tensor<64x8xf32>
}
