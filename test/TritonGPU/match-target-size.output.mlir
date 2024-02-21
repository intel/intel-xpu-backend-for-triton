// RUN: triton-opt %s -split-input-file | FileCheck %s

// CHECK: tt.dot
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %c48_i32 = arith.constant 48 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg4, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %2, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %4, %9 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.remsi %2, %7 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.extsi %arg3 : i32 to i64
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg6 : i32 to i64
    %20 = arith.divsi %1, %c4_i32 : i32
    %21 = arith.andi %20, %c7_i32 : i32
    %22 = arith.muli %21, %c32_i32 : i32
    %23 = arith.addi %22, %16 : i32
    %24 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %25 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%23, %c16_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %26 = arith.muli %15, %c128_i32 : i32
    %27 = arith.extsi %arg4 : i32 to i64
    %28 = arith.extsi %arg7 : i32 to i64
    %29 = arith.andi %1, %c3_i32 : i32
    %30 = arith.muli %29, %c64_i32 : i32
    %31 = arith.addi %30, %26 : i32
    %32 = tt.make_tensor_ptr %arg1, [%18, %27], [%28, %c1_i64], [%c0_i32, %31] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %33 = arith.addi %31, %c16_i32 : i32
    %34 = tt.make_tensor_ptr %arg1, [%18, %27], [%28, %c1_i64], [%c0_i32, %33] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %35 = arith.addi %31, %c32_i32 : i32
    %36 = tt.make_tensor_ptr %arg1, [%18, %27], [%28, %c1_i64], [%c0_i32, %35] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %37 = arith.addi %31, %c48_i32 : i32
    %38 = tt.make_tensor_ptr %arg1, [%18, %27], [%28, %c1_i64], [%c0_i32, %37] {order = array<i32: 1, 0>} : <tensor<32x16xf16>, 1>
    %39:14 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %cst, %arg12 = %cst, %arg13 = %cst, %arg14 = %cst, %arg15 = %cst, %arg16 = %cst, %arg17 = %cst, %arg18 = %24, %arg19 = %25, %arg20 = %32, %arg21 = %34, %arg22 = %36, %arg23 = %38) -> (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>)  : i32 {
      %58 = tt.load %arg18 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %59 = tt.load %arg19 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %60 = tt.load %arg20 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %61 = tt.load %arg21 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %62 = tt.load %arg22 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %63 = tt.load %arg23 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x16xf16>, 1> -> tensor<32x16xf16>
      %64 = tt.extract %arg10, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %65 = tt.extract %58, 0 : tensor<32x16xf16> -> tensor<8x16xf16>
      %66 = tt.extract %60, 0 : tensor<32x16xf16> -> tensor<16x16xf16>
      %67 = tt.dot %65, %66, %64 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %68 = tt.extract %59, 0 : tensor<32x16xf16> -> tensor<8x16xf16>
      %69 = tt.extract %60, 1 : tensor<32x16xf16> -> tensor<16x16xf16>
      %70 = tt.dot %68, %69, %67 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %71 = tt.extract %arg10, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %72 = tt.extract %58, 1 : tensor<32x16xf16> -> tensor<8x16xf16>
      %73 = tt.dot %72, %66, %71 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %74 = tt.extract %59, 1 : tensor<32x16xf16> -> tensor<8x16xf16>
      %75 = tt.dot %74, %69, %73 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %76 = tt.glue %70, %75 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %77 = tt.extract %arg11, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %78 = tt.extract %58, 2 : tensor<32x16xf16> -> tensor<8x16xf16>
      %79 = tt.dot %78, %66, %77 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %80 = tt.extract %59, 2 : tensor<32x16xf16> -> tensor<8x16xf16>
      %81 = tt.dot %80, %69, %79 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %82 = tt.extract %arg11, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %83 = tt.extract %58, 3 : tensor<32x16xf16> -> tensor<8x16xf16>
      %84 = tt.dot %83, %66, %82 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %85 = tt.extract %59, 3 : tensor<32x16xf16> -> tensor<8x16xf16>
      %86 = tt.dot %85, %69, %84 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %87 = tt.glue %81, %86 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %88 = tt.extract %arg12, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %89 = tt.extract %61, 0 : tensor<32x16xf16> -> tensor<16x16xf16>
      %90 = tt.dot %65, %89, %88 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %91 = tt.extract %61, 1 : tensor<32x16xf16> -> tensor<16x16xf16>
      %92 = tt.dot %68, %91, %90 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %93 = tt.extract %arg12, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %94 = tt.dot %72, %89, %93 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %95 = tt.dot %74, %91, %94 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %96 = tt.glue %92, %95 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %97 = tt.extract %arg13, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %98 = tt.dot %78, %89, %97 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %99 = tt.dot %80, %91, %98 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %100 = tt.extract %arg13, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %101 = tt.dot %83, %89, %100 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %102 = tt.dot %85, %91, %101 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %103 = tt.glue %99, %102 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %104 = tt.extract %arg14, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %105 = tt.extract %62, 0 : tensor<32x16xf16> -> tensor<16x16xf16>
      %106 = tt.dot %65, %105, %104 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %107 = tt.extract %62, 1 : tensor<32x16xf16> -> tensor<16x16xf16>
      %108 = tt.dot %68, %107, %106 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %109 = tt.extract %arg14, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %110 = tt.dot %72, %105, %109 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %111 = tt.dot %74, %107, %110 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %112 = tt.glue %108, %111 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %113 = tt.extract %arg15, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %114 = tt.dot %78, %105, %113 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %115 = tt.dot %80, %107, %114 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %116 = tt.extract %arg15, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %117 = tt.dot %83, %105, %116 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %118 = tt.dot %85, %107, %117 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %119 = tt.glue %115, %118 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %120 = tt.extract %arg16, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %121 = tt.extract %63, 0 : tensor<32x16xf16> -> tensor<16x16xf16>
      %122 = tt.dot %65, %121, %120 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %123 = tt.extract %63, 1 : tensor<32x16xf16> -> tensor<16x16xf16>
      %124 = tt.dot %68, %123, %122 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %125 = tt.extract %arg16, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %126 = tt.dot %72, %121, %125 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %127 = tt.dot %74, %123, %126 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %128 = tt.glue %124, %127 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %129 = tt.extract %arg17, 0 : tensor<16x16xf32> -> tensor<8x16xf32>
      %130 = tt.dot %78, %121, %129 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %131 = tt.dot %80, %123, %130 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %132 = tt.extract %arg17, 1 : tensor<16x16xf32> -> tensor<8x16xf32>
      %133 = tt.dot %83, %121, %132 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %134 = tt.dot %85, %123, %133 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %135 = tt.glue %131, %134 : tensor<8x16xf32>, tensor<8x16xf32> -> tensor<16x16xf32>
      %136 = tt.advance %arg18, [%c0_i32, %c32_i32] : <tensor<32x16xf16>, 1>
      %137 = tt.advance %arg19, [%c0_i32, %c32_i32] : <tensor<32x16xf16>, 1>
      %138 = tt.advance %arg20, [%c32_i32, %c0_i32] : <tensor<32x16xf16>, 1>
      %139 = tt.advance %arg21, [%c32_i32, %c0_i32] : <tensor<32x16xf16>, 1>
      %140 = tt.advance %arg22, [%c32_i32, %c0_i32] : <tensor<32x16xf16>, 1>
      %141 = tt.advance %arg23, [%c32_i32, %c0_i32] : <tensor<32x16xf16>, 1>
      scf.yield %76, %87, %96, %103, %112, %119, %128, %135, %136, %137, %138, %139, %140, %141 : tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>, !tt.ptr<tensor<32x16xf16>, 1>
    }
    %40 = arith.truncf %39#0 : tensor<16x16xf32> to tensor<16x16xf16>
    %41 = arith.truncf %39#1 : tensor<16x16xf32> to tensor<16x16xf16>
    %42 = arith.truncf %39#2 : tensor<16x16xf32> to tensor<16x16xf16>
    %43 = arith.truncf %39#3 : tensor<16x16xf32> to tensor<16x16xf16>
    %44 = arith.truncf %39#4 : tensor<16x16xf32> to tensor<16x16xf16>
    %45 = arith.truncf %39#5 : tensor<16x16xf32> to tensor<16x16xf16>
    %46 = arith.truncf %39#6 : tensor<16x16xf32> to tensor<16x16xf16>
    %47 = arith.truncf %39#7 : tensor<16x16xf32> to tensor<16x16xf16>
    %48 = arith.extsi %arg8 : i32 to i64
    %49 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%23, %31] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %50 = arith.addi %23, %c16_i32 : i32
    %51 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%50, %31] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %52 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%23, %33] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %53 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%50, %33] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %54 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%23, %35] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %55 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%50, %35] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %56 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%23, %37] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %57 = tt.make_tensor_ptr %arg2, [%17, %27], [%48, %c1_i64], [%50, %37] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    tt.store %49, %40 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %51, %41 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %52, %42 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %53, %43 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %54, %44 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %55, %45 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %56, %46 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.store %57, %47 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>, 1>, tensor<16x16xf16>
    tt.return
  }
}
