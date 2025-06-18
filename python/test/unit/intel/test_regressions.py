import pathlib

import triton


def test_regression_4441(device, tmp_path: pathlib.Path):
    ir = """
    #blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @triton_red_fused__softmax_backward_data_div_masked_fill_native_dropout_backward_threshold_backward_10(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg4: f64, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32) attributes {noinline = false} {
        %c64_i32 = arith.constant 64 : i32
        %cst = arith.constant dense<0.000000e+00> : tensor<64x4xf32, #blocked>
        %cst_0 = arith.constant dense<204> : tensor<1x4xi32, #blocked>
        %cst_1 = arith.constant dense<0> : tensor<64x4xi8, #blocked>
        %c4_i32 = arith.constant 4 : i32
        %c204_i32 = arith.constant 204 : i32
        %c0_i32 = arith.constant 0 : i32
        %cst_2 = arith.constant dense<1.11111116> : tensor<64x4xf32, #blocked>
        %cst_3 = arith.constant dense<41632> : tensor<64x1xi32, #blocked>
        %cst_4 = arith.constant dense<41728> : tensor<64x1xi32, #blocked>
        %cst_5 = arith.constant dense<204> : tensor<64x1xi32, #blocked>
        %cst_6 = arith.constant dense<16320> : tensor<64x1xi32, #blocked>
        %0 = tt.get_program_id x : i32
        %1 = arith.muli %0, %c64_i32 : i32
        %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %4 = tt.splat %1 : i32 -> tensor<64x1xi32, #blocked>
        %5 = arith.addi %4, %3 : tensor<64x1xi32, #blocked>
        %6 = arith.cmpi slt, %5, %cst_6 : tensor<64x1xi32, #blocked>
        %7 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
        %9 = arith.remsi %5, %cst_5 : tensor<64x1xi32, #blocked>
        %10 = arith.divsi %5, %cst_5 : tensor<64x1xi32, #blocked>
        %11 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked>
        %12 = tt.broadcast %11 : tensor<64x1xi32, #blocked> -> tensor<64x4xi32, #blocked>
        %13 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x4x!tt.ptr<f32>, #blocked>
        %14 = tt.broadcast %6 : tensor<64x1xi1, #blocked> -> tensor<64x4xi1, #blocked>
        %15 = arith.muli %9, %cst_5 : tensor<64x1xi32, #blocked>
        %16 = tt.broadcast %15 : tensor<64x1xi32, #blocked> -> tensor<64x4xi32, #blocked>
        %17 = arith.muli %10, %cst_4 : tensor<64x1xi32, #blocked>
        %18 = tt.broadcast %17 : tensor<64x1xi32, #blocked> -> tensor<64x4xi32, #blocked>
        %19 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<64x4x!tt.ptr<i1>, #blocked>
        %20 = arith.muli %10, %cst_3 : tensor<64x1xi32, #blocked>
        %21 = tt.broadcast %20 : tensor<64x1xi32, #blocked> -> tensor<64x4xi32, #blocked>
        %22 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x4x!tt.ptr<f32>, #blocked>
        %23 = scf.for %arg7 = %c0_i32 to %c204_i32 step %c4_i32 iter_args(%arg8 = %cst) -> (tensor<64x4xf32, #blocked>)  : i32 {
          %30 = tt.splat %arg7 : i32 -> tensor<1x4xi32, #blocked>
          %31 = arith.addi %30, %8 : tensor<1x4xi32, #blocked>
          %32 = arith.cmpi slt, %31, %cst_0 : tensor<1x4xi32, #blocked>
          %33 = tt.broadcast %31 : tensor<1x4xi32, #blocked> -> tensor<64x4xi32, #blocked>
          %34 = arith.addi %33, %12 : tensor<64x4xi32, #blocked>
          %35 = tt.addptr %13, %34 : tensor<64x4x!tt.ptr<f32>, #blocked>, tensor<64x4xi32, #blocked>
          %36 = tt.broadcast %32 : tensor<1x4xi1, #blocked> -> tensor<64x4xi1, #blocked>
          %37 = arith.andi %36, %14 : tensor<64x4xi1, #blocked>
          %38 = tt.load %35, %37, %cst evictionPolicy = evict_last : tensor<64x4x!tt.ptr<f32>, #blocked>
          %39 = arith.addi %33, %16 : tensor<64x4xi32, #blocked>
          %40 = arith.addi %39, %18 : tensor<64x4xi32, #blocked>
          %41 = tt.addptr %19, %40 : tensor<64x4x!tt.ptr<i1>, #blocked>, tensor<64x4xi32, #blocked>
          %42 = tt.bitcast %41 : tensor<64x4x!tt.ptr<i1>, #blocked> -> tensor<64x4x!tt.ptr<i8>, #blocked>
          %43 = tt.load %42, %37, %cst_1 evictionPolicy = evict_last : tensor<64x4x!tt.ptr<i8>, #blocked>
          %44 = arith.cmpi ne, %43, %cst_1 : tensor<64x4xi8, #blocked>
          %45 = arith.addi %39, %21 : tensor<64x4xi32, #blocked>
          %46 = tt.addptr %22, %45 : tensor<64x4x!tt.ptr<f32>, #blocked>, tensor<64x4xi32, #blocked>
          %47 = tt.load %46, %37, %cst evictionPolicy = evict_last : tensor<64x4x!tt.ptr<f32>, #blocked>
          %48 = arith.uitofp %44 : tensor<64x4xi1, #blocked> to tensor<64x4xf32, #blocked>
          %49 = arith.mulf %48, %cst_2 : tensor<64x4xf32, #blocked>
          %50 = arith.mulf %38, %49 : tensor<64x4xf32, #blocked>
          %51 = arith.mulf %50, %47 : tensor<64x4xf32, #blocked>
          %52 = arith.addf %arg8, %51 : tensor<64x4xf32, #blocked>
          %53 = arith.select %37, %52, %arg8 : tensor<64x4xi1, #blocked>, tensor<64x4xf32, #blocked>
          scf.yield %53 : tensor<64x4xf32, #blocked>
        }
        %24 = "tt.reduce"(%23) <{axis = 1 : i32}> ({
        ^bb0(%arg7: f32, %arg8: f32):
          %30 = arith.addf %arg7, %arg8 : f32
          tt.reduce.return %30 : f32
        }) : (tensor<64x4xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %25 = tt.expand_dims %24 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
        %26 = tt.splat %arg3 : !tt.ptr<i1> -> tensor<64x4x!tt.ptr<i1>, #blocked>
        %27 = tt.broadcast %25 : tensor<64x1xf32, #blocked> -> tensor<64x4xf32, #blocked>
        %28 = arith.truncf %arg4 : f64 to f32
        %29 = tt.splat %28 : f32 -> tensor<64x4xf32, #blocked>
        scf.for %arg7 = %c0_i32 to %c204_i32 step %c4_i32  : i32 {
          %30 = tt.splat %arg7 : i32 -> tensor<1x4xi32, #blocked>
          %31 = arith.addi %30, %8 : tensor<1x4xi32, #blocked>
          %32 = arith.cmpi slt, %31, %cst_0 : tensor<1x4xi32, #blocked>
          %33 = tt.broadcast %31 : tensor<1x4xi32, #blocked> -> tensor<64x4xi32, #blocked>
          %34 = arith.addi %33, %16 : tensor<64x4xi32, #blocked>
          %35 = arith.addi %34, %18 : tensor<64x4xi32, #blocked>
          %36 = tt.addptr %26, %35 : tensor<64x4x!tt.ptr<i1>, #blocked>, tensor<64x4xi32, #blocked>
          %37 = tt.broadcast %32 : tensor<1x4xi1, #blocked> -> tensor<64x4xi1, #blocked>
          %38 = arith.andi %37, %14 : tensor<64x4xi1, #blocked>
          %39 = tt.bitcast %36 : tensor<64x4x!tt.ptr<i1>, #blocked> -> tensor<64x4x!tt.ptr<i8>, #blocked>
          %40 = tt.load %39, %38, %cst_1 evictionPolicy = evict_first : tensor<64x4x!tt.ptr<i8>, #blocked>
          %41 = arith.cmpi ne, %40, %cst_1 : tensor<64x4xi8, #blocked>
          %42 = arith.addi %34, %21 : tensor<64x4xi32, #blocked>
          %43 = tt.addptr %22, %42 : tensor<64x4x!tt.ptr<f32>, #blocked>, tensor<64x4xi32, #blocked>
          %44 = tt.load %43, %38, %cst evictionPolicy = evict_first : tensor<64x4x!tt.ptr<f32>, #blocked>
          %45 = arith.addi %33, %12 : tensor<64x4xi32, #blocked>
          %46 = tt.addptr %13, %45 : tensor<64x4x!tt.ptr<f32>, #blocked>, tensor<64x4xi32, #blocked>
          %47 = tt.load %46, %38, %cst evictionPolicy = evict_first : tensor<64x4x!tt.ptr<f32>, #blocked>
          %48 = tt.addptr %19, %35 : tensor<64x4x!tt.ptr<i1>, #blocked>, tensor<64x4xi32, #blocked>
          %49 = tt.bitcast %48 : tensor<64x4x!tt.ptr<i1>, #blocked> -> tensor<64x4x!tt.ptr<i8>, #blocked>
          %50 = tt.load %49, %38, %cst_1 evictionPolicy = evict_first : tensor<64x4x!tt.ptr<i8>, #blocked>
          %51 = arith.cmpi ne, %50, %cst_1 : tensor<64x4xi8, #blocked>
          %52 = arith.subf %cst, %44 : tensor<64x4xf32, #blocked>
          %53 = arith.uitofp %51 : tensor<64x4xi1, #blocked> to tensor<64x4xf32, #blocked>
          %54 = arith.mulf %53, %cst_2 : tensor<64x4xf32, #blocked>
          %55 = arith.mulf %47, %54 : tensor<64x4xf32, #blocked>
          %56 = arith.mulf %55, %44 : tensor<64x4xf32, #blocked>
          %57 = tt.extern_elementwise %52, %27, %56 {libname = "", libpath = "", pure = true, symbol = "__imf_fmaf"} : (tensor<64x4xf32, #blocked>, tensor<64x4xf32, #blocked>, tensor<64x4xf32, #blocked>) -> tensor<64x4xf32, #blocked>
          %58 = arith.select %41, %cst, %57 : tensor<64x4xi1, #blocked>, tensor<64x4xf32, #blocked>
          %59 = arith.divf %58, %29 : tensor<64x4xf32, #blocked>
          tt.store %43, %59, %38 : tensor<64x4x!tt.ptr<f32>, #blocked>
        }
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_block_load_dpas_layout.ttgir"
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
