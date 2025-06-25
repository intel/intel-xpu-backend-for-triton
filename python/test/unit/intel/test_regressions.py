import pathlib

import triton


def test_regression_4441(device, tmp_path: pathlib.Path):
    ir = """
    #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @triton_red_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_10(%arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: f32) {
        %cst_1 = arith.constant dense<0> : tensor<64x4xi8, #blocked>
        %c4_i32 = arith.constant 4 : i32
        %c204_i32 = arith.constant 204 : i32
        %c0_i32 = arith.constant 0 : i32
        %cst_2 = arith.constant dense<1.11111116> : tensor<64x4xf32, #blocked>
        %cst_5 = arith.constant dense<204> : tensor<64x1xi32, #blocked>
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c4_i32 : i32
        %4 = tt.splat %1 : i32 -> tensor<64x1xi32, #blocked>
        %6 = arith.cmpi slt, %4, %cst_5 : tensor<64x1xi32, #blocked>
        %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x4x!tt.ptr<f32>, #blocked>
        %14 = tt.broadcast %6 : tensor<64x1xi1, #blocked> -> tensor<64x4xi1, #blocked>
        %16 = tt.broadcast %4 : tensor<64x1xi32, #blocked> -> tensor<64x4xi32, #blocked>

        %26 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<64x4x!tt.ptr<i8>, #blocked>
        %29 = tt.splat %arg4 : f32 -> tensor<64x4xf32, #blocked>
        scf.for %arg7 = %c0_i32 to %c204_i32 step %c4_i32  : i32 {
          %40 = tt.load %26, %14, %cst_1 : tensor<64x4x!tt.ptr<i8>, #blocked>
          %41 = arith.cmpi ne, %40, %cst_1 : tensor<64x4xi8, #blocked>
          %43 = tt.addptr %13, %16 : tensor<64x4x!tt.ptr<f32>, #blocked>, tensor<64x4xi32, #blocked>
          %44 = tt.load %43, %14, %cst_2 : tensor<64x4x!tt.ptr<f32>, #blocked>
          %57 = tt.extern_elementwise %44, %cst_2, %29 {libname = "", libpath = "", pure = true, symbol = "__imf_fmaf"} : (tensor<64x4xf32, #blocked>, tensor<64x4xf32, #blocked>, tensor<64x4xf32, #blocked>) -> tensor<64x4xf32, #blocked>
          %58 = arith.select %41, %cst_2, %57 : tensor<64x4xi1, #blocked>, tensor<64x4xf32, #blocked>
          %59 = arith.divf %58, %29 : tensor<64x4xf32, #blocked>
          tt.store %43, %59, %14 : tensor<64x4x!tt.ptr<f32>, #blocked>
        }
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_regression_4441.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    from triton.runtime.driver import driver
    device = driver.active.get_current_device()

    # try to catch:
    # L0 build module failed. Log: IGC: Internal Compiler Error: Segmentation violation
    # Error during Intel loadBinary: Triton Error [ZE]: 0x70000004
    # RuntimeError: Triton Error [ZE]: 0x70000004
    module, function, n_regs, n_spills, n_max_threads = driver.active.utils.load_binary(
        kernel.name, kernel.kernel, kernel.metadata.shared, kernel.metadata.build_flags,
        not kernel.metadata.generate_native_code, device)

def test_kernel_from_09_tutorial(device, tmp_path: pathlib.Path):
    # although the kernel is taken from the arl-h machine, the problem with it is also reproduced on pvc
    ir = """
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#loc = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0)
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 8 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.target_arch = "spir64"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":126:0)) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked> loc(#loc1)
    %c63_i32 = arith.constant 63 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked1> loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked2> loc(#loc1)
    %c8_i32 = arith.constant 8 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %cst_2 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc1)
    %cst_3 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc1)
    %cst_4 = arith.constant dense<64> : tensor<128x64xi32, #blocked2> loc(#loc1)
    %cst_5 = arith.constant dense<64> : tensor<64x128xi32, #blocked1> loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.addi %arg3, %c127_i32 : i32 loc(#loc58)
    %2 = arith.divsi %1, %c128_i32 : i32 loc(#loc59)
    %3 = arith.addi %arg4, %c127_i32 : i32 loc(#loc60)
    %4 = arith.divsi %3, %c128_i32 : i32 loc(#loc61)
    %5 = arith.muli %4, %c8_i32 : i32 loc(#loc7)
    %6 = arith.divsi %0, %5 : i32 loc(#loc8)
    %7 = arith.muli %6, %c8_i32 : i32 loc(#loc9)
    %8 = arith.subi %2, %7 : i32 loc(#loc10)
    %9 = arith.minsi %8, %c8_i32 : i32 loc(#loc11)
    %10 = arith.remsi %0, %9 : i32 loc(#loc12)
    %11 = arith.addi %7, %10 : i32 loc(#loc13)
    %12 = arith.remsi %0, %5 : i32 loc(#loc14)
    %13 = arith.divsi %12, %9 : i32 loc(#loc15)
    %14 = arith.muli %11, %c128_i32 : i32 loc(#loc16)
    %15 = arith.muli %13, %c128_i32 : i32 loc(#loc17)
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc18)
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> loc(#loc18)
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc18)
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> loc(#loc18)
    %20 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc19)
    %21 = tt.splat %14 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> loc(#loc19)
    %22 = arith.addi %20, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc19)
    %23 = arith.addi %21, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> loc(#loc19)
    %24 = tt.splat %15 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc20)
    %25 = tt.splat %15 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> loc(#loc20)
    %26 = arith.addi %24, %18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc20)
    %27 = arith.addi %25, %19 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> loc(#loc20)
    %28 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc21)
    %29 = arith.cmpi slt, %22, %28 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc21)
    %30 = arith.select %29, %22, %cst_2 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc22)
    %31 = tt.splat %arg4 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc23)
    %32 = arith.cmpi slt, %26, %31 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc23)
    %33 = arith.select %32, %26, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc24)
    %34 = tt.expand_dims %30 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2> loc(#loc25)
    %35 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc26)
    %36 = arith.muli %34, %35 : tensor<128x1xi32, #blocked2> loc(#loc26)
    %37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc27)
    %38 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2> loc(#loc27)
    %39 = tt.broadcast %36 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2> loc(#loc28)
    %40 = tt.broadcast %38 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2> loc(#loc28)
    %41 = arith.addi %39, %40 : tensor<128x64xi32, #blocked2> loc(#loc28)
    %42 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked2> loc(#loc29)
    %43 = tt.addptr %42, %41 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2> loc(#loc29)
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc30)
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1> loc(#loc30)
    %46 = tt.expand_dims %33 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1> loc(#loc31)
    %47 = tt.splat %arg7 : i32 -> tensor<1x128xi32, #blocked1> loc(#loc32)
    %48 = arith.muli %46, %47 : tensor<1x128xi32, #blocked1> loc(#loc32)
    %49 = tt.broadcast %45 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1> loc(#loc33)
    %50 = tt.broadcast %48 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1> loc(#loc33)
    %51 = arith.addi %49, %50 : tensor<64x128xi32, #blocked1> loc(#loc33)
    %52 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1> loc(#loc34)
    %53 = tt.addptr %52, %51 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1> loc(#loc34)
    %54 = arith.addi %arg5, %c63_i32 : i32 loc(#loc62)
    %55 = arith.divsi %54, %c64_i32 : i32 loc(#loc63)
    %56 = arith.remsi %arg5, %c64_i32 : i32 loc(#loc36)
    %57 = arith.cmpi eq, %56, %c0_i32 : i32 loc(#loc36)
    %58 = arith.cmpi sgt, %arg5, %c64_i32 : i32 loc(#loc36)
    %59 = arith.andi %57, %58 : i1 loc(#loc36)
    %60 = scf.if %59 -> (tensor<128x128xf32, #blocked>) {
      %79:3 = scf.for %arg9 = %c0_i32 to %55 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %43, %arg12 = %53) -> (tensor<128x128xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked1>)  : i32 {
        %80 = tt.load %arg11 : tensor<128x64x!tt.ptr<f16>, #blocked2> loc(#loc37)
        %81 = tt.load %arg12 : tensor<64x128x!tt.ptr<f16>, #blocked1> loc(#loc38)
        %82 = tt.fp_to_fp %80 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf32, #blocked2> loc(#loc39)
        %83 = ttg.local_alloc %82 : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared, #smem> loc(#loc39)
        %84 = ttg.local_load %83 : !ttg.memdesc<128x64xf32, #shared, #smem> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> loc(#loc39)
        %85 = tt.fp_to_fp %81 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf32, #blocked1> loc(#loc39)
        %86 = ttg.local_alloc %85 : (tensor<64x128xf32, #blocked1>) -> !ttg.memdesc<64x128xf32, #shared1, #smem> loc(#loc39)
        %87 = ttg.local_load %86 : !ttg.memdesc<64x128xf32, #shared1, #smem> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc39)
        %88 = tt.dot %84, %87, %arg10, inputPrecision = tf32 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked> loc(#loc39)
        %89 = tt.addptr %arg11, %cst_4 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2> loc(#loc40)
        %90 = tt.addptr %arg12, %cst_5 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1> loc(#loc41)
        scf.yield %88, %89, %90 : tensor<128x128xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked1> loc(#loc42)
      } loc(#loc36)
      scf.yield %79#0 : tensor<128x128xf32, #blocked> loc(#loc36)
    } else {
      %79:3 = scf.for %arg9 = %c0_i32 to %55 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %43, %arg12 = %53) -> (tensor<128x128xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked1>)  : i32 {
        %80 = arith.muli %arg9, %c64_i32 : i32 loc(#loc43)
        %81 = arith.subi %arg5, %80 : i32 loc(#loc44)
        %82 = tt.splat %81 : i32 -> tensor<1x64xi32, #blocked2> loc(#loc45)
        %83 = arith.cmpi slt, %38, %82 : tensor<1x64xi32, #blocked2> loc(#loc45)
        %84 = tt.broadcast %83 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2> loc(#loc37)
        %85 = tt.load %arg11, %84, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked2> loc(#loc37)
        %86 = tt.splat %81 : i32 -> tensor<64x1xi32, #blocked1> loc(#loc46)
        %87 = arith.cmpi slt, %45, %86 : tensor<64x1xi32, #blocked1> loc(#loc46)
        %88 = tt.broadcast %87 : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1> loc(#loc38)
        %89 = tt.load %arg12, %88, %cst_0 : tensor<64x128x!tt.ptr<f16>, #blocked1> loc(#loc38)
        %90 = tt.fp_to_fp %85 : tensor<128x64xf16, #blocked2> -> tensor<128x64xf32, #blocked2> loc(#loc39)
        %91 = ttg.local_alloc %90 : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared, #smem> loc(#loc39)
        %92 = ttg.local_load %91 : !ttg.memdesc<128x64xf32, #shared, #smem> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> loc(#loc39)
        %93 = tt.fp_to_fp %89 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf32, #blocked1> loc(#loc39)
        %94 = ttg.local_alloc %93 : (tensor<64x128xf32, #blocked1>) -> !ttg.memdesc<64x128xf32, #shared1, #smem> loc(#loc39)
        %95 = ttg.local_load %94 : !ttg.memdesc<64x128xf32, #shared1, #smem> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> loc(#loc39)
        %96 = tt.dot %92, %95, %arg10, inputPrecision = tf32 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked> loc(#loc39)
        %97 = tt.addptr %arg11, %cst_4 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2> loc(#loc40)
        %98 = tt.addptr %arg12, %cst_5 : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1> loc(#loc41)
        scf.yield %96, %97, %98 : tensor<128x128xf32, #blocked>, tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<64x128x!tt.ptr<f16>, #blocked1> loc(#loc42)
      } loc(#loc36)
      scf.yield %79#0 : tensor<128x128xf32, #blocked> loc(#loc36)
    } loc(#loc36)
    %61 = arith.truncf %60 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked> loc(#loc47)
    %62 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3> loc(#loc48)
    %63 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked3> loc(#loc49)
    %64 = arith.muli %63, %62 : tensor<128x1xi32, #blocked3> loc(#loc49)
    %65 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked3> loc(#loc50)
    %66 = tt.addptr %65, %64 : tensor<128x1x!tt.ptr<f16>, #blocked3>, tensor<128x1xi32, #blocked3> loc(#loc50)
    %67 = tt.expand_dims %27 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3> loc(#loc51)
    %68 = tt.broadcast %66 : tensor<128x1x!tt.ptr<f16>, #blocked3> -> tensor<128x128x!tt.ptr<f16>, #blocked3> loc(#loc52)
    %69 = tt.broadcast %67 : tensor<1x128xi32, #blocked3> -> tensor<128x128xi32, #blocked3> loc(#loc52)
    %70 = tt.addptr %68, %69 : tensor<128x128x!tt.ptr<f16>, #blocked3>, tensor<128x128xi32, #blocked3> loc(#loc52)
    %71 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked3> loc(#loc53)
    %72 = arith.cmpi slt, %62, %71 : tensor<128x1xi32, #blocked3> loc(#loc53)
    %73 = tt.splat %arg4 : i32 -> tensor<1x128xi32, #blocked3> loc(#loc54)
    %74 = arith.cmpi slt, %67, %73 : tensor<1x128xi32, #blocked3> loc(#loc54)
    %75 = tt.broadcast %72 : tensor<128x1xi1, #blocked3> -> tensor<128x128xi1, #blocked3> loc(#loc55)
    %76 = tt.broadcast %74 : tensor<1x128xi1, #blocked3> -> tensor<128x128xi1, #blocked3> loc(#loc55)
    %77 = arith.andi %75, %76 : tensor<128x128xi1, #blocked3> loc(#loc55)
    %78 = ttg.convert_layout %61 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked3> loc(#loc56)
    tt.store %70, %78, %77 : tensor<128x128x!tt.ptr<f16>, #blocked3> loc(#loc56)
    tt.return loc(#loc57)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":136:24)
#loc3 = loc("/home/runner/intel-xpu-backend-for-triton/python/triton/language/standard.py":40:22)
#loc4 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":137:27)
#loc5 = loc("/home/runner/intel-xpu-backend-for-triton/python/triton/language/standard.py":40:28)
#loc6 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":138:27)
#loc7 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":139:38)
#loc8 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":140:22)
#loc9 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":141:29)
#loc10 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":142:35)
#loc11 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":142:48)
#loc12 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":143:33)
#loc13 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":143:27)
#loc14 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":144:19)
#loc15 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":144:40)
#loc16 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":146:22)
#loc17 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":147:22)
#loc18 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":149:37)
#loc19 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":149:24)
#loc20 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":150:24)
#loc21 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":151:33)
#loc22 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":151:45)
#loc23 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":152:33)
#loc24 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":152:45)
#loc25 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":157:30)
#loc26 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":157:41)
#loc27 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":157:60)
#loc28 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":157:53)
#loc29 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":157:22)
#loc30 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":158:29)
#loc31 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":158:60)
#loc32 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":158:71)
#loc33 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":158:52)
#loc34 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":158:22)
#loc35 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":162:33)
#loc36 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":162:22)
#loc37 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":163:20)
#loc38 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":164:20)
#loc39 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":165:35)
#loc40 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":166:18)
#loc41 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":167:18)
#loc42 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":167:8)
#loc43 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":163:59)
#loc44 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":163:55)
#loc45 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":163:51)
#loc46 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":164:51)
#loc47 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":172:27)
#loc48 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":176:41)
#loc49 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":176:33)
#loc50 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":176:21)
#loc51 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":176:72)
#loc52 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":176:52)
#loc53 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":177:33)
#loc54 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":177:58)
#loc55 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":177:39)
#loc56 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":178:21)
#loc57 = loc("/home/runner/intel-xpu-backend-for-triton/python/tutorials/09-persistent-matmul.py":178:4)
#loc58 = loc(callsite(#loc3 at #loc4))
#loc59 = loc(callsite(#loc5 at #loc4))
#loc60 = loc(callsite(#loc3 at #loc6))
#loc61 = loc(callsite(#loc5 at #loc6))
#loc62 = loc(callsite(#loc3 at #loc35))
#loc63 = loc(callsite(#loc5 at #loc35))
    """

    temp_file = tmp_path / "test_regression_4441.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    from triton.runtime.driver import driver
    device = driver.active.get_current_device()

    # try to catch:
    # L0 build module failed. Log: IGC: Internal Compiler Error: Segmentation violation
    # Error during Intel loadBinary: Triton Error [ZE]: 0x70000004
    # RuntimeError: Triton Error [ZE]: 0x70000004
    module, function, n_regs, n_spills, n_max_threads = driver.active.utils.load_binary(
        kernel.name, kernel.kernel, kernel.metadata.shared, kernel.metadata.build_flags,
        not kernel.metadata.generate_native_code, device)
