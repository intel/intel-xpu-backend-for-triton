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
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 8 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.target_arch = "spir64"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked2>
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %cst_3 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %15 = arith.muli %13, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.splat %c128_i32 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %24 = tt.splat %15 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %24, %18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %29 = arith.cmpi slt, %20, %28 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %31 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %32 = arith.cmpi slt, %26, %31 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %33 = arith.select %32, %26, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %38 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %42 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked2>
    %44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %45 = tt.expand_dims %44 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %46 = tt.expand_dims %33 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %50 = tt.broadcast %46 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %52 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    %53 = tt.addptr %52, %50 : tensor<64x128x!tt.ptr<f32>, #blocked1>, tensor<64x128xi32, #blocked1>

    %80 = arith.muli %c0_i32, %c64_i32 : i32
    %81 = arith.subi %arg5, %80 : i32
    %82 = tt.splat %81 : i32 -> tensor<1x64xi32, #blocked2>
    %83 = arith.cmpi slt, %38, %82 : tensor<1x64xi32, #blocked2>
    %84 = tt.broadcast %83 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
    %85 = tt.load %42, %84, %cst_1 : tensor<128x64x!tt.ptr<f32>, #blocked2>
    %86 = tt.splat %81 : i32 -> tensor<64x1xi32, #blocked1>
    %87 = arith.cmpi slt, %45, %86 : tensor<64x1xi32, #blocked1>
    %88 = tt.broadcast %87 : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %89 = tt.load %53, %88, %cst_0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    %91 = ttg.local_alloc %85 : (tensor<128x64xf32, #blocked2>) -> !ttg.memdesc<128x64xf32, #shared, #smem>
    %92 = ttg.local_load %91 : !ttg.memdesc<128x64xf32, #shared, #smem> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %94 = ttg.local_alloc %89 : (tensor<64x128xf32, #blocked1>) -> !ttg.memdesc<64x128xf32, #shared1, #smem>
    %cst_test = arith.constant dense<1.11111116> : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_test2 = arith.constant dense<1.11111116> : tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %96 = tt.dot %92, %cst_test2, %cst, inputPrecision = tf32 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>

    %78 = ttg.convert_layout %96 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #blocked2>
    tt.return
  }
}
    """

    temp_file = tmp_path / "test_kernel_from_09_tutorial.ttgir"
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
