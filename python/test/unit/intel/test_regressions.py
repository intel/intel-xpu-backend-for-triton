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


def test_issue_4838(device, tmp_path: pathlib.Path):
    ir = """
#loc = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":334:0)
#loc2 = loc(unknown)
#loc42 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":413:26)
#loc54 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":427:44)
#loc58 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":429:44)
#loc61 = loc("x"(#loc))
#loc62 = loc("w"(#loc))
#loc63 = loc("b"(#loc))
#loc64 = loc("dy"(#loc))
#loc65 = loc("dx"(#loc))
#loc66 = loc("dw"(#loc))
#loc67 = loc("db"(#loc))
#loc68 = loc("rstd"(#loc))
#loc69 = loc("T"(#loc))
#loc107 = loc("b_c1"(#loc42))
#loc114 = loc(callsite(#loc2 at #loc54))
#loc116 = loc(callsite(#loc2 at #loc58))
#loc119 = loc(callsite(#loc2 at #loc107))
module {
  tt.func public @layer_norm_bwd_kernel(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x"(#loc)), %w: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("w"(#loc)), %b: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("b"(#loc)), %dy: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("dy"(#loc)), %dx: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("dx"(#loc)), %dw: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("dw"(#loc)), %db: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("db"(#loc)), %rstd: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("rstd"(#loc)), %T: i32 {tt.divisibility = 16 : i32} loc("T"(#loc))) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %cst = arith.constant dense<5.120000e+02> : tensor<64xf32> loc(#loc2)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x512xf32> loc(#loc2)
    %c0_i32 = arith.constant 0 : i32 loc(#loc2)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %c512_i64 = arith.constant 512 : i64 loc(#loc2)
    %c19_i32 = arith.constant 19 : i32 loc(#loc2)
    %m_d = arith.constant dense<512> : tensor<512xi32> loc(#loc70)
    %c512_i32 = arith.constant 512 : i32 loc(#loc2)
    %c56_i32 = arith.constant 56 : i32 loc(#loc2)
    %i_s = tt.get_program_id x : i32 loc(#loc71)
    %0 = arith.divsi %i_s, %c56_i32 : i32 loc(#loc5)
    %1 = arith.remsi %i_s, %c56_i32 : i32 loc(#loc6)
    %o_d = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32> loc(#loc72)
    %m_d_1 = arith.cmpi slt, %o_d, %m_d : tensor<512xi32> loc(#loc70)
    %b_w = arith.muli %0, %c512_i32 : i32 loc(#loc73)
    %b_w_2 = tt.addptr %w, %b_w : !tt.ptr<f32>, i32 loc(#loc74)
    %b_w_3 = tt.splat %b_w_2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc75)
    %b_w_4 = tt.addptr %b_w_3, %o_d : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc75)
    %b_w_5 = tt.load %b_w_4, %m_d_1 : tensor<512x!tt.ptr<f32>> loc(#loc76)
    %T_6 = arith.muli %1, %c19_i32 : i32 loc(#loc77)
    %T_7 = arith.addi %T_6, %c19_i32 : i32 loc(#loc78)
    %T_8 = arith.minsi %T_7, %T : i32 loc(#loc79)
    %p_x = tt.addptr %x, %b_w : !tt.ptr<f32>, i32 loc(#loc80)
    %p_x_9 = arith.extsi %T_8 : i32 to i64 loc(#loc81)
    %p_dy = tt.addptr %dy, %b_w : !tt.ptr<f32>, i32 loc(#loc82)
    %p_dx = tt.addptr %dx, %b_w : !tt.ptr<f32>, i32 loc(#loc83)
    %p_rstd = tt.addptr %rstd, %0 : !tt.ptr<f32>, i32 loc(#loc84)
    %b_xhat = tt.expand_dims %m_d_1 {axis = 0 : i32} : tensor<512xi1> -> tensor<1x512xi1> loc(#loc85)
    %b_xhat_10 = tt.broadcast %b_xhat : tensor<1x512xi1> -> tensor<64x512xi1> loc(#loc86)
    %m_t = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> loc(#loc87)
    %m_t_11 = tt.splat %T_8 : i32 -> tensor<64xi32> loc(#loc88)
    %b_wdy = tt.expand_dims %b_w_5 {axis = 0 : i32} : tensor<512xf32> -> tensor<1x512xf32> loc(#loc89)
    %b_wdy_12 = tt.broadcast %b_wdy : tensor<1x512xf32> -> tensor<64x512xf32> loc(#loc89)
    %b_db:2 = scf.for %i_t = %T_6 to %T_8 step %c64_i32 iter_args(%b_dw = %cst_0, %b_db_13 = %cst_0) -> (tensor<64x512xf32>, tensor<64x512xf32>)  : i32 {
      %p_x_14 = tt.make_tensor_ptr %p_x, [%p_x_9, %c512_i64], [%c512_i64, %c1_i64], [%i_t, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x512xf32>> loc(#loc81)
      %p_dy_15 = tt.make_tensor_ptr %p_dy, [%p_x_9, %c512_i64], [%c512_i64, %c1_i64], [%i_t, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x512xf32>> loc(#loc91)
      %p_dx_16 = tt.make_tensor_ptr %p_dx, [%p_x_9, %c512_i64], [%c512_i64, %c1_i64], [%i_t, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x512xf32>> loc(#loc92)
      %b_x = tt.load %p_x_14 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x512xf32>> loc(#loc93)
      %b_dy = tt.load %p_dy_15 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x512xf32>> loc(#loc94)
      %p_rstd_17 = tt.make_tensor_ptr %p_rstd, [%p_x_9], [%c1_i64], [%i_t] {order = array<i32: 0>} : <tensor<64xf32>> loc(#loc95)
      %b_rstd = tt.load %p_rstd_17 {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xf32>> loc(#loc96)
      %b_xhat_18 = tt.expand_dims %b_rstd {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> loc(#loc97)
      %b_xhat_19 = tt.broadcast %b_xhat_18 : tensor<64x1xf32> -> tensor<64x512xf32> loc(#loc98)
      %b_xhat_20 = arith.mulf %b_x, %b_xhat_19 : tensor<64x512xf32> loc(#loc98)
      %b_xhat_21 = arith.select %b_xhat_10, %b_xhat_20, %cst_0 : tensor<64x512xi1>, tensor<64x512xf32> loc(#loc86)
      %m_t_22 = tt.splat %i_t : i32 -> tensor<64xi32> loc(#loc99)
      %m_t_23 = arith.addi %m_t_22, %m_t : tensor<64xi32> loc(#loc99)
      %m_t_24 = arith.cmpi slt, %m_t_23, %m_t_11 : tensor<64xi32> loc(#loc88)
      %b_wdy_25 = arith.mulf %b_dy, %b_wdy_12 : tensor<64x512xf32> loc(#loc89)
      %b_dw_26 = tt.expand_dims %m_t_24 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1> loc(#loc100)
      %b_dw_27 = arith.mulf %b_dy, %b_xhat_21 : tensor<64x512xf32> loc(#loc101)
      %b_dw_28 = tt.broadcast %b_dw_26 : tensor<64x1xi1> -> tensor<64x512xi1> loc(#loc102)
      %b_dw_29 = arith.select %b_dw_28, %b_dw_27, %cst_0 : tensor<64x512xi1>, tensor<64x512xf32> loc(#loc102)
      %b_dw_30 = arith.addf %b_dw, %b_dw_29 : tensor<64x512xf32> loc(#loc103)
      %b_db_31 = arith.select %b_dw_28, %b_dy, %cst_0 : tensor<64x512xi1>, tensor<64x512xf32> loc(#loc104)
      %b_db_32 = arith.addf %b_db_13, %b_db_31 : tensor<64x512xf32> loc(#loc105)
      %b_c1 = arith.mulf %b_xhat_21, %b_wdy_25 : tensor<64x512xf32> loc(#loc106)
      %b_c1_33 = "tt.reduce"(%b_c1) <{axis = 1 : i32}> ({
      ^bb0(%b_c1_39: f32 loc(callsite(#loc2 at #loc107)), %b_c1_40: f32 loc(callsite(#loc2 at #loc107))):
        %b_c1_41 = arith.addf %b_c1_39, %b_c1_40 : f32 loc(#loc122)
        tt.reduce.return %b_c1_41 : f32 loc(#loc118)
      }) : (tensor<64x512xf32>) -> tensor<64xf32> loc(#loc118)
      %b_c1_34 = arith.divf %b_c1_33, %cst : tensor<64xf32> loc(#loc108)
      %b_dx = tt.expand_dims %b_c1_34 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> loc(#loc109)
      %b_dx_35 = tt.broadcast %b_dx : tensor<64x1xf32> -> tensor<64x512xf32> loc(#loc110)
      %b_dx_36 = arith.mulf %b_xhat_21, %b_dx_35 : tensor<64x512xf32> loc(#loc110)
      %b_dx_37 = arith.subf %b_wdy_25, %b_dx_36 : tensor<64x512xf32> loc(#loc111)
      %b_dx_38 = arith.mulf %b_dx_37, %b_xhat_19 : tensor<64x512xf32> loc(#loc112)
      tt.store %p_dx_16, %b_dx_38 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x512xf32>> loc(#loc49)
      scf.yield %b_dw_30, %b_db_32 : tensor<64x512xf32>, tensor<64x512xf32> loc(#loc50)
    } loc(#loc117)
    %2 = arith.muli %i_s, %c512_i32 : i32 loc(#loc51)
    %3 = tt.addptr %dw, %2 : !tt.ptr<f32>, i32 loc(#loc52)
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc53)
    %5 = tt.addptr %4, %o_d : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc53)
    %6 = "tt.reduce"(%b_db#0) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32 loc(callsite(#loc2 at #loc54)), %arg10: f32 loc(callsite(#loc2 at #loc54))):
      %11 = arith.addf %arg9, %arg10 : f32 loc(#loc120)
      tt.reduce.return %11 : f32 loc(#loc113)
    }) : (tensor<64x512xf32>) -> tensor<512xf32> loc(#loc113)
    tt.store %5, %6, %m_d_1 : tensor<512x!tt.ptr<f32>> loc(#loc55)
    %7 = tt.addptr %db, %2 : !tt.ptr<f32>, i32 loc(#loc56)
    %8 = tt.splat %7 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc57)
    %9 = tt.addptr %8, %o_d : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc57)
    %10 = "tt.reduce"(%b_db#1) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32 loc(callsite(#loc2 at #loc58)), %arg10: f32 loc(callsite(#loc2 at #loc58))):
      %11 = arith.addf %arg9, %arg10 : f32 loc(#loc121)
      tt.reduce.return %11 : f32 loc(#loc115)
    }) : (tensor<64x512xf32>) -> tensor<512xf32> loc(#loc115)
    tt.store %9, %10, %m_d_1 : tensor<512x!tt.ptr<f32>> loc(#loc59)
    tt.return loc(#loc60)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":375:35)
#loc3 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":366:16)
#loc4 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":362:24)
#loc5 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":363:23)
#loc6 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":363:33)
#loc7 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":365:23)
#loc8 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":368:32)
#loc9 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":368:26)
#loc10 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":368:36)
#loc11 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":368:22)
#loc12 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":374:19)
#loc13 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":374:24)
#loc14 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":374:28)
#loc15 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":376:36)
#loc16 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":376:83)
#loc17 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":377:38)
#loc18 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":378:38)
#loc19 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":386:42)
#loc20 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":390:30)
#loc21 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":390:48)
#loc22 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":402:38)
#loc23 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":402:45)
#loc24 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":404:27)
#loc25 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":377:85)
#loc26 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":378:85)
#loc27 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":380:22)
#loc28 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":381:23)
#loc29 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":386:74)
#loc30 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":387:25)
#loc31 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":389:96)
#loc32 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":389:89)
#loc33 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":402:25)
#loc34 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":405:33)
#loc35 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":405:50)
#loc36 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":405:58)
#loc37 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":405:20)
#loc38 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":407:49)
#loc39 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":407:20)
#loc40 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":413:35)
#loc41 = loc("/home/jovyan/intel-xpu-backend-for-triton/python/triton/language/standard.py":291:36)
#loc43 = loc("/home/jovyan/intel-xpu-backend-for-triton/python/triton/language/standard.py":261:15)
#loc44 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":413:52)
#loc45 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":414:42)
#loc46 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":414:37)
#loc47 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":414:28)
#loc48 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":414:54)
#loc49 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":424:23)
#loc50 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":424:8)
#loc51 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":427:28)
#loc52 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":427:22)
#loc53 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":427:32)
#loc55 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":427:37)
#loc56 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":429:22)
#loc57 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":429:32)
#loc59 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":429:37)
#loc60 = loc("/home/jovyan/intel-xpu-backend-for-triton/flash-linear-attention/fla/modules/layernorm.py":428:4)
#loc70 = loc("m_d"(#loc3))
#loc71 = loc("i_s"(#loc4))
#loc72 = loc("o_d"(#loc7))
#loc73 = loc("b_w"(#loc8))
#loc74 = loc("b_w"(#loc9))
#loc75 = loc("b_w"(#loc10))
#loc76 = loc("b_w"(#loc11))
#loc77 = loc("T"(#loc12))
#loc78 = loc("T"(#loc13))
#loc79 = loc("T"(#loc14))
#loc80 = loc("p_x"(#loc15))
#loc81 = loc("p_x"(#loc16))
#loc82 = loc("p_dy"(#loc17))
#loc83 = loc("p_dx"(#loc18))
#loc84 = loc("p_rstd"(#loc19))
#loc85 = loc("b_xhat"(#loc20))
#loc86 = loc("b_xhat"(#loc21))
#loc87 = loc("m_t"(#loc22))
#loc88 = loc("m_t"(#loc23))
#loc89 = loc("b_wdy"(#loc24))
#loc90 = loc("b_dw"(#loc1))
#loc91 = loc("p_dy"(#loc25))
#loc92 = loc("p_dx"(#loc26))
#loc93 = loc("b_x"(#loc27))
#loc94 = loc("b_dy"(#loc28))
#loc95 = loc("p_rstd"(#loc29))
#loc96 = loc("b_rstd"(#loc30))
#loc97 = loc("b_xhat"(#loc31))
#loc98 = loc("b_xhat"(#loc32))
#loc99 = loc("m_t"(#loc33))
#loc100 = loc("b_dw"(#loc34))
#loc101 = loc("b_dw"(#loc35))
#loc102 = loc("b_dw"(#loc36))
#loc103 = loc("b_dw"(#loc37))
#loc104 = loc("b_db"(#loc38))
#loc105 = loc("b_db"(#loc39))
#loc106 = loc("b_c1"(#loc40))
#loc108 = loc("b_c1"(#loc44))
#loc109 = loc("b_dx"(#loc45))
#loc110 = loc("b_dx"(#loc46))
#loc111 = loc("b_dx"(#loc47))
#loc112 = loc("b_dx"(#loc48))
#loc113 = loc(callsite(#loc41 at #loc54))
#loc115 = loc(callsite(#loc41 at #loc58))
#loc117 = loc("b_db"(#loc90))
#loc118 = loc(callsite(#loc41 at #loc107))
#loc120 = loc(callsite(#loc43 at #loc113))
#loc121 = loc(callsite(#loc43 at #loc115))
#loc122 = loc(callsite(#loc43 at #loc118))

    """
    temp_file = tmp_path / "test_regression_4838.ttir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file), options={"num_ctas": 1, "num_stages": 3, "num_warps": 4})

    from triton.runtime.driver import driver
    device = driver.active.get_current_device()

    # try to catch:
    # RuntimeError: ZE_RESULT_ERROR_INVALID_KERNEL_NAME
    module, function, n_regs, n_spills, n_max_threads = driver.active.utils.load_binary(
        kernel.name, kernel.kernel, kernel.metadata.shared, kernel.metadata.build_flags,
        not kernel.metadata.generate_native_code, device)
