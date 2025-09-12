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
