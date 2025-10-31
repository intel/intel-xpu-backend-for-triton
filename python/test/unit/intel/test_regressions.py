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


def test_regression_5374(device, tmp_path: pathlib.Path):
    ir = """
module {
  tt.func public @triton_per_fused_sort_0(%in_ptr0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %out_ptr2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %xnumel: i32, %r0_numel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<8x128xi32>
    %cst_0 = arith.constant dense<true> : tensor<8x128xi1>
    %cst_1 = arith.constant dense<1> : tensor<1x2x1xi32>
    %tmp0 = arith.constant dense<0.000000e+00> : tensor<8x128xf32>
    %cst_2 = arith.constant dense<128> : tensor<8x1xi32>
    %xmask = arith.constant dense<10> : tensor<8x1xi32>
    %c8_i32 = arith.constant 8 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_3 = arith.muli %xoffset, %c8_i32 : i32
    %xindex = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %xindex_4 = tt.expand_dims %xindex {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %xindex_5 = tt.splat %xoffset_3 : i32 -> tensor<8x1xi32>
    %xindex_6 = arith.addi %xindex_5, %xindex_4 : tensor<8x1xi32>
    %xmask_7 = arith.cmpi slt, %xindex_6, %xmask : tensor<8x1xi32>
    %r0_index = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %r0_index_8 = tt.expand_dims %r0_index {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %tmp0_9 = arith.muli %xindex_6, %cst_2 : tensor<8x1xi32>
    %tmp0_10 = tt.broadcast %r0_index_8 : tensor<1x128xi32> -> tensor<8x128xi32>
    %tmp0_11 = tt.broadcast %tmp0_9 : tensor<8x1xi32> -> tensor<8x128xi32>
    %tmp0_12 = arith.addi %tmp0_10, %tmp0_11 : tensor<8x128xi32>
    %tmp0_13 = tt.splat %in_ptr0 : !tt.ptr<f32> -> tensor<8x128x!tt.ptr<f32>>
    %tmp0_14 = tt.addptr %tmp0_13, %tmp0_12 : tensor<8x128x!tt.ptr<f32>>, tensor<8x128xi32>
    %tmp0_15 = tt.broadcast %xmask_7 : tensor<8x1xi1> -> tensor<8x128xi1>
    %tmp0_16 = tt.load %tmp0_14, %tmp0_15, %tmp0 : tensor<8x128x!tt.ptr<f32>>
    %tmp2 = arith.trunci %r0_index_8 : tensor<1x128xi32> to tensor<1x128xi16>
    %tmp4 = tt.broadcast %tmp2 : tensor<1x128xi16> -> tensor<8x128xi16>
    %flip = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %flip_17 = tt.expand_dims %flip {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %flip_18 = tt.expand_dims %flip_17 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
    %flip_19 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<256x2x2xi32>
    %flip_20 = tt.reshape %flip_19 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %y = tt.reshape %tmp0_16 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy = tt.bitcast %y : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %left_mask = arith.subi %cst_1, %flip_18 : tensor<1x2x1xi32>
    %ileft = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<512x2x1xi32>
    %ileft_21 = arith.muli %iy, %ileft : tensor<512x2x1xi32>
    %ileft_22 = "tt.reduce"(%ileft_21) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_23 = tt.expand_dims %ileft_22 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_24 = tt.broadcast %ileft_23 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<512x2x1xi32>
    %iright_25 = arith.muli %iy, %iright : tensor<512x2x1xi32>
    %iright_26 = "tt.reduce"(%iright_25) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_27 = tt.expand_dims %iright_26 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_28 = tt.broadcast %iright_27 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_29 = tt.reshape %ileft_24 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_30 = tt.reshape %iright_28 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left = tt.bitcast %ileft_29 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right = tt.bitcast %iright_30 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx = tt.reshape %tmp4 : tensor<8x128xi16> -> tensor<512x2x1xi16>
    %left_idx = arith.trunci %left_mask : tensor<1x2x1xi32> to tensor<1x2x1xi16>
    %left_idx_31 = tt.broadcast %left_idx : tensor<1x2x1xi16> -> tensor<512x2x1xi16>
    %left_idx_32 = arith.muli %y_idx, %left_idx_31 : tensor<512x2x1xi16>
    %input = arith.extsi %left_idx_32 : tensor<512x2x1xi16> to tensor<512x2x1xi32>
    %left_idx_33 = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_34 = tt.expand_dims %left_idx_33 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_35 = tt.broadcast %left_idx_34 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx = arith.trunci %flip_18 : tensor<1x2x1xi32> to tensor<1x2x1xi16>
    %right_idx_36 = tt.broadcast %right_idx : tensor<1x2x1xi16> -> tensor<512x2x1xi16>
    %right_idx_37 = arith.muli %y_idx, %right_idx_36 : tensor<512x2x1xi16>
    %input_38 = arith.extsi %right_idx_37 : tensor<512x2x1xi16> to tensor<512x2x1xi32>
    %right_idx_39 = "tt.reduce"(%input_38) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_40 = tt.expand_dims %right_idx_39 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_41 = tt.broadcast %right_idx_40 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_42 = tt.reshape %left_idx_35 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_43 = tt.reshape %right_idx_41 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix = tt.bitcast %tmp0_16 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan = arith.cmpf une, %left, %left : tensor<8x128xf32>
    %right_isnan = arith.cmpf une, %right, %right : tensor<8x128xf32>
    %cond = arith.cmpf ogt, %left, %right : tensor<8x128xf32>
    %cond_44 = arith.xori %right_isnan, %cst_0 : tensor<8x128xi1>
    %cond_45 = arith.andi %left_isnan, %cond_44 : tensor<8x128xi1>
    %cond_46 = arith.ori %cond, %cond_45 : tensor<8x128xi1>
    %eq = arith.cmpf oeq, %left, %right : tensor<8x128xf32>
    %eq_47 = arith.andi %left_isnan, %right_isnan : tensor<8x128xi1>
    %eq_48 = arith.ori %eq, %eq_47 : tensor<8x128xi1>
    %cond_49 = arith.cmpi sgt, %left_idx_42, %right_idx_43 : tensor<8x128xi32>
    %cond_50 = arith.andi %eq_48, %cond_49 : tensor<8x128xi1>
    %cond_51 = arith.ori %cond_46, %cond_50 : tensor<8x128xi1>
    %cond_52 = arith.extui %cond_51 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_53 = arith.xori %cond_52, %flip_20 : tensor<8x128xi32>
    %cond_54 = arith.cmpi ne, %cond_53, %cst : tensor<8x128xi32>
    %ret = arith.xori %ileft_29, %iright_30 : tensor<8x128xi32>
    %ret_55 = arith.select %cond_54, %ret, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_56 = arith.xori %ix, %ret_55 : tensor<8x128xi32>
    %new_idxs = arith.xori %left_idx_42, %right_idx_43 : tensor<8x128xi32>
    %new_idxs_57 = arith.select %cond_54, %new_idxs, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_58 = arith.extsi %tmp2 : tensor<1x128xi16> to tensor<1x128xi32>
    %new_idxs_59 = tt.broadcast %new_idxs_58 : tensor<1x128xi32> -> tensor<8x128xi32>
    %new_idxs_60 = arith.xori %new_idxs_59, %new_idxs_57 : tensor<8x128xi32>
    %0 = tt.bitcast %ret_56 : tensor<8x128xi32> -> tensor<8x128xf32>
    %flip_61 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<128x2x4xi32>
    %flip_62 = tt.reshape %flip_61 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %y_63 = tt.reshape %0 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_64 = tt.bitcast %y_63 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_65 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<256x2x2xi32>
    %ileft_66 = arith.muli %iy_64, %ileft_65 : tensor<256x2x2xi32>
    %ileft_67 = "tt.reduce"(%ileft_66) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_68 = tt.expand_dims %ileft_67 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_69 = tt.broadcast %ileft_68 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_70 = arith.muli %iy_64, %flip_19 : tensor<256x2x2xi32>
    %iright_71 = "tt.reduce"(%iright_70) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_72 = tt.expand_dims %iright_71 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_73 = tt.broadcast %iright_72 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_74 = tt.reshape %ileft_69 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_75 = tt.reshape %iright_73 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_76 = tt.bitcast %ileft_74 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_77 = tt.bitcast %iright_75 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_78 = tt.reshape %new_idxs_60 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_79 = arith.muli %y_idx_78, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_80 = "tt.reduce"(%left_idx_79) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_81 = tt.expand_dims %left_idx_80 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_82 = tt.broadcast %left_idx_81 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_83 = arith.muli %y_idx_78, %flip_19 : tensor<256x2x2xi32>
    %right_idx_84 = "tt.reduce"(%right_idx_83) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_85 = tt.expand_dims %right_idx_84 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_86 = tt.broadcast %right_idx_85 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_87 = tt.reshape %left_idx_82 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_88 = tt.reshape %right_idx_86 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_89 = tt.bitcast %0 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_90 = arith.cmpf une, %left_76, %left_76 : tensor<8x128xf32>
    %right_isnan_91 = arith.cmpf une, %right_77, %right_77 : tensor<8x128xf32>
    %cond_92 = arith.cmpf ogt, %left_76, %right_77 : tensor<8x128xf32>
    %cond_93 = arith.xori %right_isnan_91, %cst_0 : tensor<8x128xi1>
    %cond_94 = arith.andi %left_isnan_90, %cond_93 : tensor<8x128xi1>
    %cond_95 = arith.ori %cond_92, %cond_94 : tensor<8x128xi1>
    %eq_96 = arith.cmpf oeq, %left_76, %right_77 : tensor<8x128xf32>
    %eq_97 = arith.andi %left_isnan_90, %right_isnan_91 : tensor<8x128xi1>
    %eq_98 = arith.ori %eq_96, %eq_97 : tensor<8x128xi1>
    %cond_99 = arith.cmpi sgt, %left_idx_87, %right_idx_88 : tensor<8x128xi32>
    %cond_100 = arith.andi %eq_98, %cond_99 : tensor<8x128xi1>
    %cond_101 = arith.ori %cond_95, %cond_100 : tensor<8x128xi1>
    %cond_102 = arith.extui %cond_101 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_103 = arith.xori %cond_102, %flip_62 : tensor<8x128xi32>
    %cond_104 = arith.cmpi ne, %cond_103, %cst : tensor<8x128xi32>
    %ret_105 = arith.xori %ileft_74, %iright_75 : tensor<8x128xi32>
    %ret_106 = arith.select %cond_104, %ret_105, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_107 = arith.xori %ix_89, %ret_106 : tensor<8x128xi32>
    %new_idxs_108 = arith.xori %left_idx_87, %right_idx_88 : tensor<8x128xi32>
    %new_idxs_109 = arith.select %cond_104, %new_idxs_108, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_110 = arith.xori %new_idxs_60, %new_idxs_109 : tensor<8x128xi32>
    %1 = tt.bitcast %ret_107 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_111 = tt.reshape %1 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_112 = tt.bitcast %y_111 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_113 = arith.muli %iy_112, %ileft : tensor<512x2x1xi32>
    %ileft_114 = "tt.reduce"(%ileft_113) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_115 = tt.expand_dims %ileft_114 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_116 = tt.broadcast %ileft_115 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_117 = arith.muli %iy_112, %iright : tensor<512x2x1xi32>
    %iright_118 = "tt.reduce"(%iright_117) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_119 = tt.expand_dims %iright_118 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_120 = tt.broadcast %iright_119 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_121 = tt.reshape %ileft_116 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_122 = tt.reshape %iright_120 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_123 = tt.bitcast %ileft_121 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_124 = tt.bitcast %iright_122 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_125 = tt.reshape %new_idxs_110 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_126 = arith.muli %y_idx_125, %ileft : tensor<512x2x1xi32>
    %left_idx_127 = "tt.reduce"(%left_idx_126) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_128 = tt.expand_dims %left_idx_127 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_129 = tt.broadcast %left_idx_128 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_130 = arith.muli %y_idx_125, %iright : tensor<512x2x1xi32>
    %right_idx_131 = "tt.reduce"(%right_idx_130) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_132 = tt.expand_dims %right_idx_131 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_133 = tt.broadcast %right_idx_132 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_134 = tt.reshape %left_idx_129 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_135 = tt.reshape %right_idx_133 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_136 = tt.bitcast %1 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_137 = arith.cmpf une, %left_123, %left_123 : tensor<8x128xf32>
    %right_isnan_138 = arith.cmpf une, %right_124, %right_124 : tensor<8x128xf32>
    %cond_139 = arith.cmpf ogt, %left_123, %right_124 : tensor<8x128xf32>
    %cond_140 = arith.xori %right_isnan_138, %cst_0 : tensor<8x128xi1>
    %cond_141 = arith.andi %left_isnan_137, %cond_140 : tensor<8x128xi1>
    %cond_142 = arith.ori %cond_139, %cond_141 : tensor<8x128xi1>
    %eq_143 = arith.cmpf oeq, %left_123, %right_124 : tensor<8x128xf32>
    %eq_144 = arith.andi %left_isnan_137, %right_isnan_138 : tensor<8x128xi1>
    %eq_145 = arith.ori %eq_143, %eq_144 : tensor<8x128xi1>
    %cond_146 = arith.cmpi sgt, %left_idx_134, %right_idx_135 : tensor<8x128xi32>
    %cond_147 = arith.andi %eq_145, %cond_146 : tensor<8x128xi1>
    %cond_148 = arith.ori %cond_142, %cond_147 : tensor<8x128xi1>
    %cond_149 = arith.extui %cond_148 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_150 = arith.xori %cond_149, %flip_62 : tensor<8x128xi32>
    %cond_151 = arith.cmpi ne, %cond_150, %cst : tensor<8x128xi32>
    %ret_152 = arith.xori %ileft_121, %iright_122 : tensor<8x128xi32>
    %ret_153 = arith.select %cond_151, %ret_152, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_154 = arith.xori %ix_136, %ret_153 : tensor<8x128xi32>
    %new_idxs_155 = arith.xori %left_idx_134, %right_idx_135 : tensor<8x128xi32>
    %new_idxs_156 = arith.select %cond_151, %new_idxs_155, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_157 = arith.xori %new_idxs_110, %new_idxs_156 : tensor<8x128xi32>
    %2 = tt.bitcast %ret_154 : tensor<8x128xi32> -> tensor<8x128xf32>
    %flip_158 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<64x2x8xi32>
    %flip_159 = tt.reshape %flip_158 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %y_160 = tt.reshape %2 : tensor<8x128xf32> -> tensor<128x2x4xf32>
    %iy_161 = tt.bitcast %y_160 : tensor<128x2x4xf32> -> tensor<128x2x4xi32>
    %ileft_162 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<128x2x4xi32>
    %ileft_163 = arith.muli %iy_161, %ileft_162 : tensor<128x2x4xi32>
    %ileft_164 = "tt.reduce"(%ileft_163) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %ileft_165 = tt.expand_dims %ileft_164 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %ileft_166 = tt.broadcast %ileft_165 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %iright_167 = arith.muli %iy_161, %flip_61 : tensor<128x2x4xi32>
    %iright_168 = "tt.reduce"(%iright_167) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %iright_169 = tt.expand_dims %iright_168 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %iright_170 = tt.broadcast %iright_169 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %ileft_171 = tt.reshape %ileft_166 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %iright_172 = tt.reshape %iright_170 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %left_173 = tt.bitcast %ileft_171 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_174 = tt.bitcast %iright_172 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_175 = tt.reshape %new_idxs_157 : tensor<8x128xi32> -> tensor<128x2x4xi32>
    %left_idx_176 = arith.muli %y_idx_175, %ileft_162 : tensor<128x2x4xi32>
    %left_idx_177 = "tt.reduce"(%left_idx_176) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %left_idx_178 = tt.expand_dims %left_idx_177 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %left_idx_179 = tt.broadcast %left_idx_178 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %right_idx_180 = arith.muli %y_idx_175, %flip_61 : tensor<128x2x4xi32>
    %right_idx_181 = "tt.reduce"(%right_idx_180) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %right_idx_182 = tt.expand_dims %right_idx_181 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %right_idx_183 = tt.broadcast %right_idx_182 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %left_idx_184 = tt.reshape %left_idx_179 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %right_idx_185 = tt.reshape %right_idx_183 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %ix_186 = tt.bitcast %2 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_187 = arith.cmpf une, %left_173, %left_173 : tensor<8x128xf32>
    %right_isnan_188 = arith.cmpf une, %right_174, %right_174 : tensor<8x128xf32>
    %cond_189 = arith.cmpf ogt, %left_173, %right_174 : tensor<8x128xf32>
    %cond_190 = arith.xori %right_isnan_188, %cst_0 : tensor<8x128xi1>
    %cond_191 = arith.andi %left_isnan_187, %cond_190 : tensor<8x128xi1>
    %cond_192 = arith.ori %cond_189, %cond_191 : tensor<8x128xi1>
    %eq_193 = arith.cmpf oeq, %left_173, %right_174 : tensor<8x128xf32>
    %eq_194 = arith.andi %left_isnan_187, %right_isnan_188 : tensor<8x128xi1>
    %eq_195 = arith.ori %eq_193, %eq_194 : tensor<8x128xi1>
    %cond_196 = arith.cmpi sgt, %left_idx_184, %right_idx_185 : tensor<8x128xi32>
    %cond_197 = arith.andi %eq_195, %cond_196 : tensor<8x128xi1>
    %cond_198 = arith.ori %cond_192, %cond_197 : tensor<8x128xi1>
    %cond_199 = arith.extui %cond_198 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_200 = arith.xori %cond_199, %flip_159 : tensor<8x128xi32>
    %cond_201 = arith.cmpi ne, %cond_200, %cst : tensor<8x128xi32>
    %ret_202 = arith.xori %ileft_171, %iright_172 : tensor<8x128xi32>
    %ret_203 = arith.select %cond_201, %ret_202, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_204 = arith.xori %ix_186, %ret_203 : tensor<8x128xi32>
    %new_idxs_205 = arith.xori %left_idx_184, %right_idx_185 : tensor<8x128xi32>
    %new_idxs_206 = arith.select %cond_201, %new_idxs_205, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_207 = arith.xori %new_idxs_157, %new_idxs_206 : tensor<8x128xi32>
    %3 = tt.bitcast %ret_204 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_208 = tt.reshape %3 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_209 = tt.bitcast %y_208 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_210 = arith.muli %iy_209, %ileft_65 : tensor<256x2x2xi32>
    %ileft_211 = "tt.reduce"(%ileft_210) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_212 = tt.expand_dims %ileft_211 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_213 = tt.broadcast %ileft_212 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_214 = arith.muli %iy_209, %flip_19 : tensor<256x2x2xi32>
    %iright_215 = "tt.reduce"(%iright_214) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_216 = tt.expand_dims %iright_215 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_217 = tt.broadcast %iright_216 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_218 = tt.reshape %ileft_213 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_219 = tt.reshape %iright_217 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_220 = tt.bitcast %ileft_218 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_221 = tt.bitcast %iright_219 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_222 = tt.reshape %new_idxs_207 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_223 = arith.muli %y_idx_222, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_224 = "tt.reduce"(%left_idx_223) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_225 = tt.expand_dims %left_idx_224 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_226 = tt.broadcast %left_idx_225 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_227 = arith.muli %y_idx_222, %flip_19 : tensor<256x2x2xi32>
    %right_idx_228 = "tt.reduce"(%right_idx_227) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_229 = tt.expand_dims %right_idx_228 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_230 = tt.broadcast %right_idx_229 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_231 = tt.reshape %left_idx_226 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_232 = tt.reshape %right_idx_230 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_233 = tt.bitcast %3 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_234 = arith.cmpf une, %left_220, %left_220 : tensor<8x128xf32>
    %right_isnan_235 = arith.cmpf une, %right_221, %right_221 : tensor<8x128xf32>
    %cond_236 = arith.cmpf ogt, %left_220, %right_221 : tensor<8x128xf32>
    %cond_237 = arith.xori %right_isnan_235, %cst_0 : tensor<8x128xi1>
    %cond_238 = arith.andi %left_isnan_234, %cond_237 : tensor<8x128xi1>
    %cond_239 = arith.ori %cond_236, %cond_238 : tensor<8x128xi1>
    %eq_240 = arith.cmpf oeq, %left_220, %right_221 : tensor<8x128xf32>
    %eq_241 = arith.andi %left_isnan_234, %right_isnan_235 : tensor<8x128xi1>
    %eq_242 = arith.ori %eq_240, %eq_241 : tensor<8x128xi1>
    %cond_243 = arith.cmpi sgt, %left_idx_231, %right_idx_232 : tensor<8x128xi32>
    %cond_244 = arith.andi %eq_242, %cond_243 : tensor<8x128xi1>
    %cond_245 = arith.ori %cond_239, %cond_244 : tensor<8x128xi1>
    %cond_246 = arith.extui %cond_245 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_247 = arith.xori %cond_246, %flip_159 : tensor<8x128xi32>
    %cond_248 = arith.cmpi ne, %cond_247, %cst : tensor<8x128xi32>
    %ret_249 = arith.xori %ileft_218, %iright_219 : tensor<8x128xi32>
    %ret_250 = arith.select %cond_248, %ret_249, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_251 = arith.xori %ix_233, %ret_250 : tensor<8x128xi32>
    %new_idxs_252 = arith.xori %left_idx_231, %right_idx_232 : tensor<8x128xi32>
    %new_idxs_253 = arith.select %cond_248, %new_idxs_252, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_254 = arith.xori %new_idxs_207, %new_idxs_253 : tensor<8x128xi32>
    %4 = tt.bitcast %ret_251 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_255 = tt.reshape %4 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_256 = tt.bitcast %y_255 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_257 = arith.muli %iy_256, %ileft : tensor<512x2x1xi32>
    %ileft_258 = "tt.reduce"(%ileft_257) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_259 = tt.expand_dims %ileft_258 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_260 = tt.broadcast %ileft_259 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_261 = arith.muli %iy_256, %iright : tensor<512x2x1xi32>
    %iright_262 = "tt.reduce"(%iright_261) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_263 = tt.expand_dims %iright_262 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_264 = tt.broadcast %iright_263 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_265 = tt.reshape %ileft_260 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_266 = tt.reshape %iright_264 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_267 = tt.bitcast %ileft_265 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_268 = tt.bitcast %iright_266 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_269 = tt.reshape %new_idxs_254 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_270 = arith.muli %y_idx_269, %ileft : tensor<512x2x1xi32>
    %left_idx_271 = "tt.reduce"(%left_idx_270) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_272 = tt.expand_dims %left_idx_271 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_273 = tt.broadcast %left_idx_272 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_274 = arith.muli %y_idx_269, %iright : tensor<512x2x1xi32>
    %right_idx_275 = "tt.reduce"(%right_idx_274) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_276 = tt.expand_dims %right_idx_275 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_277 = tt.broadcast %right_idx_276 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_278 = tt.reshape %left_idx_273 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_279 = tt.reshape %right_idx_277 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_280 = tt.bitcast %4 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_281 = arith.cmpf une, %left_267, %left_267 : tensor<8x128xf32>
    %right_isnan_282 = arith.cmpf une, %right_268, %right_268 : tensor<8x128xf32>
    %cond_283 = arith.cmpf ogt, %left_267, %right_268 : tensor<8x128xf32>
    %cond_284 = arith.xori %right_isnan_282, %cst_0 : tensor<8x128xi1>
    %cond_285 = arith.andi %left_isnan_281, %cond_284 : tensor<8x128xi1>
    %cond_286 = arith.ori %cond_283, %cond_285 : tensor<8x128xi1>
    %eq_287 = arith.cmpf oeq, %left_267, %right_268 : tensor<8x128xf32>
    %eq_288 = arith.andi %left_isnan_281, %right_isnan_282 : tensor<8x128xi1>
    %eq_289 = arith.ori %eq_287, %eq_288 : tensor<8x128xi1>
    %cond_290 = arith.cmpi sgt, %left_idx_278, %right_idx_279 : tensor<8x128xi32>
    %cond_291 = arith.andi %eq_289, %cond_290 : tensor<8x128xi1>
    %cond_292 = arith.ori %cond_286, %cond_291 : tensor<8x128xi1>
    %cond_293 = arith.extui %cond_292 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_294 = arith.xori %cond_293, %flip_159 : tensor<8x128xi32>
    %cond_295 = arith.cmpi ne, %cond_294, %cst : tensor<8x128xi32>
    %ret_296 = arith.xori %ileft_265, %iright_266 : tensor<8x128xi32>
    %ret_297 = arith.select %cond_295, %ret_296, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_298 = arith.xori %ix_280, %ret_297 : tensor<8x128xi32>
    %new_idxs_299 = arith.xori %left_idx_278, %right_idx_279 : tensor<8x128xi32>
    %new_idxs_300 = arith.select %cond_295, %new_idxs_299, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_301 = arith.xori %new_idxs_254, %new_idxs_300 : tensor<8x128xi32>
    %5 = tt.bitcast %ret_298 : tensor<8x128xi32> -> tensor<8x128xf32>
    %flip_302 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<32x2x16xi32>
    %flip_303 = tt.reshape %flip_302 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %y_304 = tt.reshape %5 : tensor<8x128xf32> -> tensor<64x2x8xf32>
    %iy_305 = tt.bitcast %y_304 : tensor<64x2x8xf32> -> tensor<64x2x8xi32>
    %ileft_306 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<64x2x8xi32>
    %ileft_307 = arith.muli %iy_305, %ileft_306 : tensor<64x2x8xi32>
    %ileft_308 = "tt.reduce"(%ileft_307) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %ileft_309 = tt.expand_dims %ileft_308 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %ileft_310 = tt.broadcast %ileft_309 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %iright_311 = arith.muli %iy_305, %flip_158 : tensor<64x2x8xi32>
    %iright_312 = "tt.reduce"(%iright_311) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %iright_313 = tt.expand_dims %iright_312 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %iright_314 = tt.broadcast %iright_313 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %ileft_315 = tt.reshape %ileft_310 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %iright_316 = tt.reshape %iright_314 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %left_317 = tt.bitcast %ileft_315 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_318 = tt.bitcast %iright_316 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_319 = tt.reshape %new_idxs_301 : tensor<8x128xi32> -> tensor<64x2x8xi32>
    %left_idx_320 = arith.muli %y_idx_319, %ileft_306 : tensor<64x2x8xi32>
    %left_idx_321 = "tt.reduce"(%left_idx_320) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %left_idx_322 = tt.expand_dims %left_idx_321 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %left_idx_323 = tt.broadcast %left_idx_322 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %right_idx_324 = arith.muli %y_idx_319, %flip_158 : tensor<64x2x8xi32>
    %right_idx_325 = "tt.reduce"(%right_idx_324) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %right_idx_326 = tt.expand_dims %right_idx_325 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %right_idx_327 = tt.broadcast %right_idx_326 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %left_idx_328 = tt.reshape %left_idx_323 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %right_idx_329 = tt.reshape %right_idx_327 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %ix_330 = tt.bitcast %5 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_331 = arith.cmpf une, %left_317, %left_317 : tensor<8x128xf32>
    %right_isnan_332 = arith.cmpf une, %right_318, %right_318 : tensor<8x128xf32>
    %cond_333 = arith.cmpf ogt, %left_317, %right_318 : tensor<8x128xf32>
    %cond_334 = arith.xori %right_isnan_332, %cst_0 : tensor<8x128xi1>
    %cond_335 = arith.andi %left_isnan_331, %cond_334 : tensor<8x128xi1>
    %cond_336 = arith.ori %cond_333, %cond_335 : tensor<8x128xi1>
    %eq_337 = arith.cmpf oeq, %left_317, %right_318 : tensor<8x128xf32>
    %eq_338 = arith.andi %left_isnan_331, %right_isnan_332 : tensor<8x128xi1>
    %eq_339 = arith.ori %eq_337, %eq_338 : tensor<8x128xi1>
    %cond_340 = arith.cmpi sgt, %left_idx_328, %right_idx_329 : tensor<8x128xi32>
    %cond_341 = arith.andi %eq_339, %cond_340 : tensor<8x128xi1>
    %cond_342 = arith.ori %cond_336, %cond_341 : tensor<8x128xi1>
    %cond_343 = arith.extui %cond_342 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_344 = arith.xori %cond_343, %flip_303 : tensor<8x128xi32>
    %cond_345 = arith.cmpi ne, %cond_344, %cst : tensor<8x128xi32>
    %ret_346 = arith.xori %ileft_315, %iright_316 : tensor<8x128xi32>
    %ret_347 = arith.select %cond_345, %ret_346, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_348 = arith.xori %ix_330, %ret_347 : tensor<8x128xi32>
    %new_idxs_349 = arith.xori %left_idx_328, %right_idx_329 : tensor<8x128xi32>
    %new_idxs_350 = arith.select %cond_345, %new_idxs_349, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_351 = arith.xori %new_idxs_301, %new_idxs_350 : tensor<8x128xi32>
    %6 = tt.bitcast %ret_348 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_352 = tt.reshape %6 : tensor<8x128xf32> -> tensor<128x2x4xf32>
    %iy_353 = tt.bitcast %y_352 : tensor<128x2x4xf32> -> tensor<128x2x4xi32>
    %ileft_354 = arith.muli %iy_353, %ileft_162 : tensor<128x2x4xi32>
    %ileft_355 = "tt.reduce"(%ileft_354) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %ileft_356 = tt.expand_dims %ileft_355 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %ileft_357 = tt.broadcast %ileft_356 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %iright_358 = arith.muli %iy_353, %flip_61 : tensor<128x2x4xi32>
    %iright_359 = "tt.reduce"(%iright_358) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %iright_360 = tt.expand_dims %iright_359 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %iright_361 = tt.broadcast %iright_360 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %ileft_362 = tt.reshape %ileft_357 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %iright_363 = tt.reshape %iright_361 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %left_364 = tt.bitcast %ileft_362 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_365 = tt.bitcast %iright_363 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_366 = tt.reshape %new_idxs_351 : tensor<8x128xi32> -> tensor<128x2x4xi32>
    %left_idx_367 = arith.muli %y_idx_366, %ileft_162 : tensor<128x2x4xi32>
    %left_idx_368 = "tt.reduce"(%left_idx_367) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %left_idx_369 = tt.expand_dims %left_idx_368 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %left_idx_370 = tt.broadcast %left_idx_369 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %right_idx_371 = arith.muli %y_idx_366, %flip_61 : tensor<128x2x4xi32>
    %right_idx_372 = "tt.reduce"(%right_idx_371) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %right_idx_373 = tt.expand_dims %right_idx_372 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %right_idx_374 = tt.broadcast %right_idx_373 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %left_idx_375 = tt.reshape %left_idx_370 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %right_idx_376 = tt.reshape %right_idx_374 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %ix_377 = tt.bitcast %6 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_378 = arith.cmpf une, %left_364, %left_364 : tensor<8x128xf32>
    %right_isnan_379 = arith.cmpf une, %right_365, %right_365 : tensor<8x128xf32>
    %cond_380 = arith.cmpf ogt, %left_364, %right_365 : tensor<8x128xf32>
    %cond_381 = arith.xori %right_isnan_379, %cst_0 : tensor<8x128xi1>
    %cond_382 = arith.andi %left_isnan_378, %cond_381 : tensor<8x128xi1>
    %cond_383 = arith.ori %cond_380, %cond_382 : tensor<8x128xi1>
    %eq_384 = arith.cmpf oeq, %left_364, %right_365 : tensor<8x128xf32>
    %eq_385 = arith.andi %left_isnan_378, %right_isnan_379 : tensor<8x128xi1>
    %eq_386 = arith.ori %eq_384, %eq_385 : tensor<8x128xi1>
    %cond_387 = arith.cmpi sgt, %left_idx_375, %right_idx_376 : tensor<8x128xi32>
    %cond_388 = arith.andi %eq_386, %cond_387 : tensor<8x128xi1>
    %cond_389 = arith.ori %cond_383, %cond_388 : tensor<8x128xi1>
    %cond_390 = arith.extui %cond_389 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_391 = arith.xori %cond_390, %flip_303 : tensor<8x128xi32>
    %cond_392 = arith.cmpi ne, %cond_391, %cst : tensor<8x128xi32>
    %ret_393 = arith.xori %ileft_362, %iright_363 : tensor<8x128xi32>
    %ret_394 = arith.select %cond_392, %ret_393, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_395 = arith.xori %ix_377, %ret_394 : tensor<8x128xi32>
    %new_idxs_396 = arith.xori %left_idx_375, %right_idx_376 : tensor<8x128xi32>
    %new_idxs_397 = arith.select %cond_392, %new_idxs_396, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_398 = arith.xori %new_idxs_351, %new_idxs_397 : tensor<8x128xi32>
    %7 = tt.bitcast %ret_395 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_399 = tt.reshape %7 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_400 = tt.bitcast %y_399 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_401 = arith.muli %iy_400, %ileft_65 : tensor<256x2x2xi32>
    %ileft_402 = "tt.reduce"(%ileft_401) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_403 = tt.expand_dims %ileft_402 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_404 = tt.broadcast %ileft_403 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_405 = arith.muli %iy_400, %flip_19 : tensor<256x2x2xi32>
    %iright_406 = "tt.reduce"(%iright_405) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_407 = tt.expand_dims %iright_406 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_408 = tt.broadcast %iright_407 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_409 = tt.reshape %ileft_404 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_410 = tt.reshape %iright_408 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_411 = tt.bitcast %ileft_409 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_412 = tt.bitcast %iright_410 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_413 = tt.reshape %new_idxs_398 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_414 = arith.muli %y_idx_413, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_415 = "tt.reduce"(%left_idx_414) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_416 = tt.expand_dims %left_idx_415 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_417 = tt.broadcast %left_idx_416 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_418 = arith.muli %y_idx_413, %flip_19 : tensor<256x2x2xi32>
    %right_idx_419 = "tt.reduce"(%right_idx_418) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_420 = tt.expand_dims %right_idx_419 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_421 = tt.broadcast %right_idx_420 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_422 = tt.reshape %left_idx_417 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_423 = tt.reshape %right_idx_421 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_424 = tt.bitcast %7 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_425 = arith.cmpf une, %left_411, %left_411 : tensor<8x128xf32>
    %right_isnan_426 = arith.cmpf une, %right_412, %right_412 : tensor<8x128xf32>
    %cond_427 = arith.cmpf ogt, %left_411, %right_412 : tensor<8x128xf32>
    %cond_428 = arith.xori %right_isnan_426, %cst_0 : tensor<8x128xi1>
    %cond_429 = arith.andi %left_isnan_425, %cond_428 : tensor<8x128xi1>
    %cond_430 = arith.ori %cond_427, %cond_429 : tensor<8x128xi1>
    %eq_431 = arith.cmpf oeq, %left_411, %right_412 : tensor<8x128xf32>
    %eq_432 = arith.andi %left_isnan_425, %right_isnan_426 : tensor<8x128xi1>
    %eq_433 = arith.ori %eq_431, %eq_432 : tensor<8x128xi1>
    %cond_434 = arith.cmpi sgt, %left_idx_422, %right_idx_423 : tensor<8x128xi32>
    %cond_435 = arith.andi %eq_433, %cond_434 : tensor<8x128xi1>
    %cond_436 = arith.ori %cond_430, %cond_435 : tensor<8x128xi1>
    %cond_437 = arith.extui %cond_436 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_438 = arith.xori %cond_437, %flip_303 : tensor<8x128xi32>
    %cond_439 = arith.cmpi ne, %cond_438, %cst : tensor<8x128xi32>
    %ret_440 = arith.xori %ileft_409, %iright_410 : tensor<8x128xi32>
    %ret_441 = arith.select %cond_439, %ret_440, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_442 = arith.xori %ix_424, %ret_441 : tensor<8x128xi32>
    %new_idxs_443 = arith.xori %left_idx_422, %right_idx_423 : tensor<8x128xi32>
    %new_idxs_444 = arith.select %cond_439, %new_idxs_443, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_445 = arith.xori %new_idxs_398, %new_idxs_444 : tensor<8x128xi32>
    %8 = tt.bitcast %ret_442 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_446 = tt.reshape %8 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_447 = tt.bitcast %y_446 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_448 = arith.muli %iy_447, %ileft : tensor<512x2x1xi32>
    %ileft_449 = "tt.reduce"(%ileft_448) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_450 = tt.expand_dims %ileft_449 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_451 = tt.broadcast %ileft_450 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_452 = arith.muli %iy_447, %iright : tensor<512x2x1xi32>
    %iright_453 = "tt.reduce"(%iright_452) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_454 = tt.expand_dims %iright_453 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_455 = tt.broadcast %iright_454 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_456 = tt.reshape %ileft_451 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_457 = tt.reshape %iright_455 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_458 = tt.bitcast %ileft_456 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_459 = tt.bitcast %iright_457 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_460 = tt.reshape %new_idxs_445 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_461 = arith.muli %y_idx_460, %ileft : tensor<512x2x1xi32>
    %left_idx_462 = "tt.reduce"(%left_idx_461) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_463 = tt.expand_dims %left_idx_462 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_464 = tt.broadcast %left_idx_463 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_465 = arith.muli %y_idx_460, %iright : tensor<512x2x1xi32>
    %right_idx_466 = "tt.reduce"(%right_idx_465) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_467 = tt.expand_dims %right_idx_466 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_468 = tt.broadcast %right_idx_467 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_469 = tt.reshape %left_idx_464 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_470 = tt.reshape %right_idx_468 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_471 = tt.bitcast %8 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_472 = arith.cmpf une, %left_458, %left_458 : tensor<8x128xf32>
    %right_isnan_473 = arith.cmpf une, %right_459, %right_459 : tensor<8x128xf32>
    %cond_474 = arith.cmpf ogt, %left_458, %right_459 : tensor<8x128xf32>
    %cond_475 = arith.xori %right_isnan_473, %cst_0 : tensor<8x128xi1>
    %cond_476 = arith.andi %left_isnan_472, %cond_475 : tensor<8x128xi1>
    %cond_477 = arith.ori %cond_474, %cond_476 : tensor<8x128xi1>
    %eq_478 = arith.cmpf oeq, %left_458, %right_459 : tensor<8x128xf32>
    %eq_479 = arith.andi %left_isnan_472, %right_isnan_473 : tensor<8x128xi1>
    %eq_480 = arith.ori %eq_478, %eq_479 : tensor<8x128xi1>
    %cond_481 = arith.cmpi sgt, %left_idx_469, %right_idx_470 : tensor<8x128xi32>
    %cond_482 = arith.andi %eq_480, %cond_481 : tensor<8x128xi1>
    %cond_483 = arith.ori %cond_477, %cond_482 : tensor<8x128xi1>
    %cond_484 = arith.extui %cond_483 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_485 = arith.xori %cond_484, %flip_303 : tensor<8x128xi32>
    %cond_486 = arith.cmpi ne, %cond_485, %cst : tensor<8x128xi32>
    %ret_487 = arith.xori %ileft_456, %iright_457 : tensor<8x128xi32>
    %ret_488 = arith.select %cond_486, %ret_487, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_489 = arith.xori %ix_471, %ret_488 : tensor<8x128xi32>
    %new_idxs_490 = arith.xori %left_idx_469, %right_idx_470 : tensor<8x128xi32>
    %new_idxs_491 = arith.select %cond_486, %new_idxs_490, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_492 = arith.xori %new_idxs_445, %new_idxs_491 : tensor<8x128xi32>
    %9 = tt.bitcast %ret_489 : tensor<8x128xi32> -> tensor<8x128xf32>
    %flip_493 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<16x2x32xi32>
    %flip_494 = tt.reshape %flip_493 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %y_495 = tt.reshape %9 : tensor<8x128xf32> -> tensor<32x2x16xf32>
    %iy_496 = tt.bitcast %y_495 : tensor<32x2x16xf32> -> tensor<32x2x16xi32>
    %ileft_497 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<32x2x16xi32>
    %ileft_498 = arith.muli %iy_496, %ileft_497 : tensor<32x2x16xi32>
    %ileft_499 = "tt.reduce"(%ileft_498) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %ileft_500 = tt.expand_dims %ileft_499 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %ileft_501 = tt.broadcast %ileft_500 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %iright_502 = arith.muli %iy_496, %flip_302 : tensor<32x2x16xi32>
    %iright_503 = "tt.reduce"(%iright_502) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %iright_504 = tt.expand_dims %iright_503 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %iright_505 = tt.broadcast %iright_504 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %ileft_506 = tt.reshape %ileft_501 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %iright_507 = tt.reshape %iright_505 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %left_508 = tt.bitcast %ileft_506 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_509 = tt.bitcast %iright_507 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_510 = tt.reshape %new_idxs_492 : tensor<8x128xi32> -> tensor<32x2x16xi32>
    %left_idx_511 = arith.muli %y_idx_510, %ileft_497 : tensor<32x2x16xi32>
    %left_idx_512 = "tt.reduce"(%left_idx_511) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %left_idx_513 = tt.expand_dims %left_idx_512 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %left_idx_514 = tt.broadcast %left_idx_513 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %right_idx_515 = arith.muli %y_idx_510, %flip_302 : tensor<32x2x16xi32>
    %right_idx_516 = "tt.reduce"(%right_idx_515) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %right_idx_517 = tt.expand_dims %right_idx_516 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %right_idx_518 = tt.broadcast %right_idx_517 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %left_idx_519 = tt.reshape %left_idx_514 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %right_idx_520 = tt.reshape %right_idx_518 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %ix_521 = tt.bitcast %9 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_522 = arith.cmpf une, %left_508, %left_508 : tensor<8x128xf32>
    %right_isnan_523 = arith.cmpf une, %right_509, %right_509 : tensor<8x128xf32>
    %cond_524 = arith.cmpf ogt, %left_508, %right_509 : tensor<8x128xf32>
    %cond_525 = arith.xori %right_isnan_523, %cst_0 : tensor<8x128xi1>
    %cond_526 = arith.andi %left_isnan_522, %cond_525 : tensor<8x128xi1>
    %cond_527 = arith.ori %cond_524, %cond_526 : tensor<8x128xi1>
    %eq_528 = arith.cmpf oeq, %left_508, %right_509 : tensor<8x128xf32>
    %eq_529 = arith.andi %left_isnan_522, %right_isnan_523 : tensor<8x128xi1>
    %eq_530 = arith.ori %eq_528, %eq_529 : tensor<8x128xi1>
    %cond_531 = arith.cmpi sgt, %left_idx_519, %right_idx_520 : tensor<8x128xi32>
    %cond_532 = arith.andi %eq_530, %cond_531 : tensor<8x128xi1>
    %cond_533 = arith.ori %cond_527, %cond_532 : tensor<8x128xi1>
    %cond_534 = arith.extui %cond_533 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_535 = arith.xori %cond_534, %flip_494 : tensor<8x128xi32>
    %cond_536 = arith.cmpi ne, %cond_535, %cst : tensor<8x128xi32>
    %ret_537 = arith.xori %ileft_506, %iright_507 : tensor<8x128xi32>
    %ret_538 = arith.select %cond_536, %ret_537, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_539 = arith.xori %ix_521, %ret_538 : tensor<8x128xi32>
    %new_idxs_540 = arith.xori %left_idx_519, %right_idx_520 : tensor<8x128xi32>
    %new_idxs_541 = arith.select %cond_536, %new_idxs_540, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_542 = arith.xori %new_idxs_492, %new_idxs_541 : tensor<8x128xi32>
    %10 = tt.bitcast %ret_539 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_543 = tt.reshape %10 : tensor<8x128xf32> -> tensor<64x2x8xf32>
    %iy_544 = tt.bitcast %y_543 : tensor<64x2x8xf32> -> tensor<64x2x8xi32>
    %ileft_545 = arith.muli %iy_544, %ileft_306 : tensor<64x2x8xi32>
    %ileft_546 = "tt.reduce"(%ileft_545) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %ileft_547 = tt.expand_dims %ileft_546 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %ileft_548 = tt.broadcast %ileft_547 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %iright_549 = arith.muli %iy_544, %flip_158 : tensor<64x2x8xi32>
    %iright_550 = "tt.reduce"(%iright_549) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %iright_551 = tt.expand_dims %iright_550 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %iright_552 = tt.broadcast %iright_551 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %ileft_553 = tt.reshape %ileft_548 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %iright_554 = tt.reshape %iright_552 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %left_555 = tt.bitcast %ileft_553 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_556 = tt.bitcast %iright_554 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_557 = tt.reshape %new_idxs_542 : tensor<8x128xi32> -> tensor<64x2x8xi32>
    %left_idx_558 = arith.muli %y_idx_557, %ileft_306 : tensor<64x2x8xi32>
    %left_idx_559 = "tt.reduce"(%left_idx_558) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %left_idx_560 = tt.expand_dims %left_idx_559 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %left_idx_561 = tt.broadcast %left_idx_560 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %right_idx_562 = arith.muli %y_idx_557, %flip_158 : tensor<64x2x8xi32>
    %right_idx_563 = "tt.reduce"(%right_idx_562) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %right_idx_564 = tt.expand_dims %right_idx_563 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %right_idx_565 = tt.broadcast %right_idx_564 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %left_idx_566 = tt.reshape %left_idx_561 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %right_idx_567 = tt.reshape %right_idx_565 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %ix_568 = tt.bitcast %10 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_569 = arith.cmpf une, %left_555, %left_555 : tensor<8x128xf32>
    %right_isnan_570 = arith.cmpf une, %right_556, %right_556 : tensor<8x128xf32>
    %cond_571 = arith.cmpf ogt, %left_555, %right_556 : tensor<8x128xf32>
    %cond_572 = arith.xori %right_isnan_570, %cst_0 : tensor<8x128xi1>
    %cond_573 = arith.andi %left_isnan_569, %cond_572 : tensor<8x128xi1>
    %cond_574 = arith.ori %cond_571, %cond_573 : tensor<8x128xi1>
    %eq_575 = arith.cmpf oeq, %left_555, %right_556 : tensor<8x128xf32>
    %eq_576 = arith.andi %left_isnan_569, %right_isnan_570 : tensor<8x128xi1>
    %eq_577 = arith.ori %eq_575, %eq_576 : tensor<8x128xi1>
    %cond_578 = arith.cmpi sgt, %left_idx_566, %right_idx_567 : tensor<8x128xi32>
    %cond_579 = arith.andi %eq_577, %cond_578 : tensor<8x128xi1>
    %cond_580 = arith.ori %cond_574, %cond_579 : tensor<8x128xi1>
    %cond_581 = arith.extui %cond_580 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_582 = arith.xori %cond_581, %flip_494 : tensor<8x128xi32>
    %cond_583 = arith.cmpi ne, %cond_582, %cst : tensor<8x128xi32>
    %ret_584 = arith.xori %ileft_553, %iright_554 : tensor<8x128xi32>
    %ret_585 = arith.select %cond_583, %ret_584, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_586 = arith.xori %ix_568, %ret_585 : tensor<8x128xi32>
    %new_idxs_587 = arith.xori %left_idx_566, %right_idx_567 : tensor<8x128xi32>
    %new_idxs_588 = arith.select %cond_583, %new_idxs_587, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_589 = arith.xori %new_idxs_542, %new_idxs_588 : tensor<8x128xi32>
    %11 = tt.bitcast %ret_586 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_590 = tt.reshape %11 : tensor<8x128xf32> -> tensor<128x2x4xf32>
    %iy_591 = tt.bitcast %y_590 : tensor<128x2x4xf32> -> tensor<128x2x4xi32>
    %ileft_592 = arith.muli %iy_591, %ileft_162 : tensor<128x2x4xi32>
    %ileft_593 = "tt.reduce"(%ileft_592) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %ileft_594 = tt.expand_dims %ileft_593 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %ileft_595 = tt.broadcast %ileft_594 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %iright_596 = arith.muli %iy_591, %flip_61 : tensor<128x2x4xi32>
    %iright_597 = "tt.reduce"(%iright_596) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %iright_598 = tt.expand_dims %iright_597 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %iright_599 = tt.broadcast %iright_598 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %ileft_600 = tt.reshape %ileft_595 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %iright_601 = tt.reshape %iright_599 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %left_602 = tt.bitcast %ileft_600 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_603 = tt.bitcast %iright_601 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_604 = tt.reshape %new_idxs_589 : tensor<8x128xi32> -> tensor<128x2x4xi32>
    %left_idx_605 = arith.muli %y_idx_604, %ileft_162 : tensor<128x2x4xi32>
    %left_idx_606 = "tt.reduce"(%left_idx_605) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %left_idx_607 = tt.expand_dims %left_idx_606 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %left_idx_608 = tt.broadcast %left_idx_607 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %right_idx_609 = arith.muli %y_idx_604, %flip_61 : tensor<128x2x4xi32>
    %right_idx_610 = "tt.reduce"(%right_idx_609) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %right_idx_611 = tt.expand_dims %right_idx_610 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %right_idx_612 = tt.broadcast %right_idx_611 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %left_idx_613 = tt.reshape %left_idx_608 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %right_idx_614 = tt.reshape %right_idx_612 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %ix_615 = tt.bitcast %11 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_616 = arith.cmpf une, %left_602, %left_602 : tensor<8x128xf32>
    %right_isnan_617 = arith.cmpf une, %right_603, %right_603 : tensor<8x128xf32>
    %cond_618 = arith.cmpf ogt, %left_602, %right_603 : tensor<8x128xf32>
    %cond_619 = arith.xori %right_isnan_617, %cst_0 : tensor<8x128xi1>
    %cond_620 = arith.andi %left_isnan_616, %cond_619 : tensor<8x128xi1>
    %cond_621 = arith.ori %cond_618, %cond_620 : tensor<8x128xi1>
    %eq_622 = arith.cmpf oeq, %left_602, %right_603 : tensor<8x128xf32>
    %eq_623 = arith.andi %left_isnan_616, %right_isnan_617 : tensor<8x128xi1>
    %eq_624 = arith.ori %eq_622, %eq_623 : tensor<8x128xi1>
    %cond_625 = arith.cmpi sgt, %left_idx_613, %right_idx_614 : tensor<8x128xi32>
    %cond_626 = arith.andi %eq_624, %cond_625 : tensor<8x128xi1>
    %cond_627 = arith.ori %cond_621, %cond_626 : tensor<8x128xi1>
    %cond_628 = arith.extui %cond_627 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_629 = arith.xori %cond_628, %flip_494 : tensor<8x128xi32>
    %cond_630 = arith.cmpi ne, %cond_629, %cst : tensor<8x128xi32>
    %ret_631 = arith.xori %ileft_600, %iright_601 : tensor<8x128xi32>
    %ret_632 = arith.select %cond_630, %ret_631, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_633 = arith.xori %ix_615, %ret_632 : tensor<8x128xi32>
    %new_idxs_634 = arith.xori %left_idx_613, %right_idx_614 : tensor<8x128xi32>
    %new_idxs_635 = arith.select %cond_630, %new_idxs_634, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_636 = arith.xori %new_idxs_589, %new_idxs_635 : tensor<8x128xi32>
    %12 = tt.bitcast %ret_633 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_637 = tt.reshape %12 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_638 = tt.bitcast %y_637 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_639 = arith.muli %iy_638, %ileft_65 : tensor<256x2x2xi32>
    %ileft_640 = "tt.reduce"(%ileft_639) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_641 = tt.expand_dims %ileft_640 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_642 = tt.broadcast %ileft_641 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_643 = arith.muli %iy_638, %flip_19 : tensor<256x2x2xi32>
    %iright_644 = "tt.reduce"(%iright_643) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_645 = tt.expand_dims %iright_644 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_646 = tt.broadcast %iright_645 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_647 = tt.reshape %ileft_642 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_648 = tt.reshape %iright_646 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_649 = tt.bitcast %ileft_647 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_650 = tt.bitcast %iright_648 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_651 = tt.reshape %new_idxs_636 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_652 = arith.muli %y_idx_651, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_653 = "tt.reduce"(%left_idx_652) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_654 = tt.expand_dims %left_idx_653 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_655 = tt.broadcast %left_idx_654 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_656 = arith.muli %y_idx_651, %flip_19 : tensor<256x2x2xi32>
    %right_idx_657 = "tt.reduce"(%right_idx_656) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_658 = tt.expand_dims %right_idx_657 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_659 = tt.broadcast %right_idx_658 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_660 = tt.reshape %left_idx_655 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_661 = tt.reshape %right_idx_659 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_662 = tt.bitcast %12 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_663 = arith.cmpf une, %left_649, %left_649 : tensor<8x128xf32>
    %right_isnan_664 = arith.cmpf une, %right_650, %right_650 : tensor<8x128xf32>
    %cond_665 = arith.cmpf ogt, %left_649, %right_650 : tensor<8x128xf32>
    %cond_666 = arith.xori %right_isnan_664, %cst_0 : tensor<8x128xi1>
    %cond_667 = arith.andi %left_isnan_663, %cond_666 : tensor<8x128xi1>
    %cond_668 = arith.ori %cond_665, %cond_667 : tensor<8x128xi1>
    %eq_669 = arith.cmpf oeq, %left_649, %right_650 : tensor<8x128xf32>
    %eq_670 = arith.andi %left_isnan_663, %right_isnan_664 : tensor<8x128xi1>
    %eq_671 = arith.ori %eq_669, %eq_670 : tensor<8x128xi1>
    %cond_672 = arith.cmpi sgt, %left_idx_660, %right_idx_661 : tensor<8x128xi32>
    %cond_673 = arith.andi %eq_671, %cond_672 : tensor<8x128xi1>
    %cond_674 = arith.ori %cond_668, %cond_673 : tensor<8x128xi1>
    %cond_675 = arith.extui %cond_674 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_676 = arith.xori %cond_675, %flip_494 : tensor<8x128xi32>
    %cond_677 = arith.cmpi ne, %cond_676, %cst : tensor<8x128xi32>
    %ret_678 = arith.xori %ileft_647, %iright_648 : tensor<8x128xi32>
    %ret_679 = arith.select %cond_677, %ret_678, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_680 = arith.xori %ix_662, %ret_679 : tensor<8x128xi32>
    %new_idxs_681 = arith.xori %left_idx_660, %right_idx_661 : tensor<8x128xi32>
    %new_idxs_682 = arith.select %cond_677, %new_idxs_681, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_683 = arith.xori %new_idxs_636, %new_idxs_682 : tensor<8x128xi32>
    %13 = tt.bitcast %ret_680 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_684 = tt.reshape %13 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_685 = tt.bitcast %y_684 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_686 = arith.muli %iy_685, %ileft : tensor<512x2x1xi32>
    %ileft_687 = "tt.reduce"(%ileft_686) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_688 = tt.expand_dims %ileft_687 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_689 = tt.broadcast %ileft_688 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_690 = arith.muli %iy_685, %iright : tensor<512x2x1xi32>
    %iright_691 = "tt.reduce"(%iright_690) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_692 = tt.expand_dims %iright_691 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_693 = tt.broadcast %iright_692 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_694 = tt.reshape %ileft_689 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_695 = tt.reshape %iright_693 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_696 = tt.bitcast %ileft_694 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_697 = tt.bitcast %iright_695 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_698 = tt.reshape %new_idxs_683 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_699 = arith.muli %y_idx_698, %ileft : tensor<512x2x1xi32>
    %left_idx_700 = "tt.reduce"(%left_idx_699) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_701 = tt.expand_dims %left_idx_700 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_702 = tt.broadcast %left_idx_701 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_703 = arith.muli %y_idx_698, %iright : tensor<512x2x1xi32>
    %right_idx_704 = "tt.reduce"(%right_idx_703) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_705 = tt.expand_dims %right_idx_704 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_706 = tt.broadcast %right_idx_705 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_707 = tt.reshape %left_idx_702 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_708 = tt.reshape %right_idx_706 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_709 = tt.bitcast %13 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_710 = arith.cmpf une, %left_696, %left_696 : tensor<8x128xf32>
    %right_isnan_711 = arith.cmpf une, %right_697, %right_697 : tensor<8x128xf32>
    %cond_712 = arith.cmpf ogt, %left_696, %right_697 : tensor<8x128xf32>
    %cond_713 = arith.xori %right_isnan_711, %cst_0 : tensor<8x128xi1>
    %cond_714 = arith.andi %left_isnan_710, %cond_713 : tensor<8x128xi1>
    %cond_715 = arith.ori %cond_712, %cond_714 : tensor<8x128xi1>
    %eq_716 = arith.cmpf oeq, %left_696, %right_697 : tensor<8x128xf32>
    %eq_717 = arith.andi %left_isnan_710, %right_isnan_711 : tensor<8x128xi1>
    %eq_718 = arith.ori %eq_716, %eq_717 : tensor<8x128xi1>
    %cond_719 = arith.cmpi sgt, %left_idx_707, %right_idx_708 : tensor<8x128xi32>
    %cond_720 = arith.andi %eq_718, %cond_719 : tensor<8x128xi1>
    %cond_721 = arith.ori %cond_715, %cond_720 : tensor<8x128xi1>
    %cond_722 = arith.extui %cond_721 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_723 = arith.xori %cond_722, %flip_494 : tensor<8x128xi32>
    %cond_724 = arith.cmpi ne, %cond_723, %cst : tensor<8x128xi32>
    %ret_725 = arith.xori %ileft_694, %iright_695 : tensor<8x128xi32>
    %ret_726 = arith.select %cond_724, %ret_725, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_727 = arith.xori %ix_709, %ret_726 : tensor<8x128xi32>
    %new_idxs_728 = arith.xori %left_idx_707, %right_idx_708 : tensor<8x128xi32>
    %new_idxs_729 = arith.select %cond_724, %new_idxs_728, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_730 = arith.xori %new_idxs_683, %new_idxs_729 : tensor<8x128xi32>
    %14 = tt.bitcast %ret_727 : tensor<8x128xi32> -> tensor<8x128xf32>
    %flip_731 = tt.broadcast %flip_18 : tensor<1x2x1xi32> -> tensor<8x2x64xi32>
    %flip_732 = tt.reshape %flip_731 : tensor<8x2x64xi32> -> tensor<8x128xi32>
    %y_733 = tt.reshape %14 : tensor<8x128xf32> -> tensor<16x2x32xf32>
    %iy_734 = tt.bitcast %y_733 : tensor<16x2x32xf32> -> tensor<16x2x32xi32>
    %ileft_735 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<16x2x32xi32>
    %ileft_736 = arith.muli %iy_734, %ileft_735 : tensor<16x2x32xi32>
    %ileft_737 = "tt.reduce"(%ileft_736) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %ileft_738 = tt.expand_dims %ileft_737 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %ileft_739 = tt.broadcast %ileft_738 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %iright_740 = arith.muli %iy_734, %flip_493 : tensor<16x2x32xi32>
    %iright_741 = "tt.reduce"(%iright_740) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %iright_742 = tt.expand_dims %iright_741 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %iright_743 = tt.broadcast %iright_742 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %ileft_744 = tt.reshape %ileft_739 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %iright_745 = tt.reshape %iright_743 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %left_746 = tt.bitcast %ileft_744 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_747 = tt.bitcast %iright_745 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_748 = tt.reshape %new_idxs_730 : tensor<8x128xi32> -> tensor<16x2x32xi32>
    %left_idx_749 = arith.muli %y_idx_748, %ileft_735 : tensor<16x2x32xi32>
    %left_idx_750 = "tt.reduce"(%left_idx_749) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %left_idx_751 = tt.expand_dims %left_idx_750 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %left_idx_752 = tt.broadcast %left_idx_751 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %right_idx_753 = arith.muli %y_idx_748, %flip_493 : tensor<16x2x32xi32>
    %right_idx_754 = "tt.reduce"(%right_idx_753) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %right_idx_755 = tt.expand_dims %right_idx_754 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %right_idx_756 = tt.broadcast %right_idx_755 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %left_idx_757 = tt.reshape %left_idx_752 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %right_idx_758 = tt.reshape %right_idx_756 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %ix_759 = tt.bitcast %14 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_760 = arith.cmpf une, %left_746, %left_746 : tensor<8x128xf32>
    %right_isnan_761 = arith.cmpf une, %right_747, %right_747 : tensor<8x128xf32>
    %cond_762 = arith.cmpf ogt, %left_746, %right_747 : tensor<8x128xf32>
    %cond_763 = arith.xori %right_isnan_761, %cst_0 : tensor<8x128xi1>
    %cond_764 = arith.andi %left_isnan_760, %cond_763 : tensor<8x128xi1>
    %cond_765 = arith.ori %cond_762, %cond_764 : tensor<8x128xi1>
    %eq_766 = arith.cmpf oeq, %left_746, %right_747 : tensor<8x128xf32>
    %eq_767 = arith.andi %left_isnan_760, %right_isnan_761 : tensor<8x128xi1>
    %eq_768 = arith.ori %eq_766, %eq_767 : tensor<8x128xi1>
    %cond_769 = arith.cmpi sgt, %left_idx_757, %right_idx_758 : tensor<8x128xi32>
    %cond_770 = arith.andi %eq_768, %cond_769 : tensor<8x128xi1>
    %cond_771 = arith.ori %cond_765, %cond_770 : tensor<8x128xi1>
    %cond_772 = arith.extui %cond_771 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_773 = arith.xori %cond_772, %flip_732 : tensor<8x128xi32>
    %cond_774 = arith.cmpi ne, %cond_773, %cst : tensor<8x128xi32>
    %ret_775 = arith.xori %ileft_744, %iright_745 : tensor<8x128xi32>
    %ret_776 = arith.select %cond_774, %ret_775, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_777 = arith.xori %ix_759, %ret_776 : tensor<8x128xi32>
    %new_idxs_778 = arith.xori %left_idx_757, %right_idx_758 : tensor<8x128xi32>
    %new_idxs_779 = arith.select %cond_774, %new_idxs_778, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_780 = arith.xori %new_idxs_730, %new_idxs_779 : tensor<8x128xi32>
    %15 = tt.bitcast %ret_777 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_781 = tt.reshape %15 : tensor<8x128xf32> -> tensor<32x2x16xf32>
    %iy_782 = tt.bitcast %y_781 : tensor<32x2x16xf32> -> tensor<32x2x16xi32>
    %ileft_783 = arith.muli %iy_782, %ileft_497 : tensor<32x2x16xi32>
    %ileft_784 = "tt.reduce"(%ileft_783) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %ileft_785 = tt.expand_dims %ileft_784 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %ileft_786 = tt.broadcast %ileft_785 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %iright_787 = arith.muli %iy_782, %flip_302 : tensor<32x2x16xi32>
    %iright_788 = "tt.reduce"(%iright_787) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %iright_789 = tt.expand_dims %iright_788 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %iright_790 = tt.broadcast %iright_789 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %ileft_791 = tt.reshape %ileft_786 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %iright_792 = tt.reshape %iright_790 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %left_793 = tt.bitcast %ileft_791 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_794 = tt.bitcast %iright_792 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_795 = tt.reshape %new_idxs_780 : tensor<8x128xi32> -> tensor<32x2x16xi32>
    %left_idx_796 = arith.muli %y_idx_795, %ileft_497 : tensor<32x2x16xi32>
    %left_idx_797 = "tt.reduce"(%left_idx_796) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %left_idx_798 = tt.expand_dims %left_idx_797 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %left_idx_799 = tt.broadcast %left_idx_798 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %right_idx_800 = arith.muli %y_idx_795, %flip_302 : tensor<32x2x16xi32>
    %right_idx_801 = "tt.reduce"(%right_idx_800) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %right_idx_802 = tt.expand_dims %right_idx_801 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %right_idx_803 = tt.broadcast %right_idx_802 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %left_idx_804 = tt.reshape %left_idx_799 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %right_idx_805 = tt.reshape %right_idx_803 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %ix_806 = tt.bitcast %15 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_807 = arith.cmpf une, %left_793, %left_793 : tensor<8x128xf32>
    %right_isnan_808 = arith.cmpf une, %right_794, %right_794 : tensor<8x128xf32>
    %cond_809 = arith.cmpf ogt, %left_793, %right_794 : tensor<8x128xf32>
    %cond_810 = arith.xori %right_isnan_808, %cst_0 : tensor<8x128xi1>
    %cond_811 = arith.andi %left_isnan_807, %cond_810 : tensor<8x128xi1>
    %cond_812 = arith.ori %cond_809, %cond_811 : tensor<8x128xi1>
    %eq_813 = arith.cmpf oeq, %left_793, %right_794 : tensor<8x128xf32>
    %eq_814 = arith.andi %left_isnan_807, %right_isnan_808 : tensor<8x128xi1>
    %eq_815 = arith.ori %eq_813, %eq_814 : tensor<8x128xi1>
    %cond_816 = arith.cmpi sgt, %left_idx_804, %right_idx_805 : tensor<8x128xi32>
    %cond_817 = arith.andi %eq_815, %cond_816 : tensor<8x128xi1>
    %cond_818 = arith.ori %cond_812, %cond_817 : tensor<8x128xi1>
    %cond_819 = arith.extui %cond_818 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_820 = arith.xori %cond_819, %flip_732 : tensor<8x128xi32>
    %cond_821 = arith.cmpi ne, %cond_820, %cst : tensor<8x128xi32>
    %ret_822 = arith.xori %ileft_791, %iright_792 : tensor<8x128xi32>
    %ret_823 = arith.select %cond_821, %ret_822, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_824 = arith.xori %ix_806, %ret_823 : tensor<8x128xi32>
    %new_idxs_825 = arith.xori %left_idx_804, %right_idx_805 : tensor<8x128xi32>
    %new_idxs_826 = arith.select %cond_821, %new_idxs_825, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_827 = arith.xori %new_idxs_780, %new_idxs_826 : tensor<8x128xi32>
    %16 = tt.bitcast %ret_824 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_828 = tt.reshape %16 : tensor<8x128xf32> -> tensor<64x2x8xf32>
    %iy_829 = tt.bitcast %y_828 : tensor<64x2x8xf32> -> tensor<64x2x8xi32>
    %ileft_830 = arith.muli %iy_829, %ileft_306 : tensor<64x2x8xi32>
    %ileft_831 = "tt.reduce"(%ileft_830) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %ileft_832 = tt.expand_dims %ileft_831 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %ileft_833 = tt.broadcast %ileft_832 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %iright_834 = arith.muli %iy_829, %flip_158 : tensor<64x2x8xi32>
    %iright_835 = "tt.reduce"(%iright_834) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %iright_836 = tt.expand_dims %iright_835 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %iright_837 = tt.broadcast %iright_836 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %ileft_838 = tt.reshape %ileft_833 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %iright_839 = tt.reshape %iright_837 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %left_840 = tt.bitcast %ileft_838 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_841 = tt.bitcast %iright_839 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_842 = tt.reshape %new_idxs_827 : tensor<8x128xi32> -> tensor<64x2x8xi32>
    %left_idx_843 = arith.muli %y_idx_842, %ileft_306 : tensor<64x2x8xi32>
    %left_idx_844 = "tt.reduce"(%left_idx_843) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %left_idx_845 = tt.expand_dims %left_idx_844 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %left_idx_846 = tt.broadcast %left_idx_845 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %right_idx_847 = arith.muli %y_idx_842, %flip_158 : tensor<64x2x8xi32>
    %right_idx_848 = "tt.reduce"(%right_idx_847) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %right_idx_849 = tt.expand_dims %right_idx_848 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %right_idx_850 = tt.broadcast %right_idx_849 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %left_idx_851 = tt.reshape %left_idx_846 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %right_idx_852 = tt.reshape %right_idx_850 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %ix_853 = tt.bitcast %16 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_854 = arith.cmpf une, %left_840, %left_840 : tensor<8x128xf32>
    %right_isnan_855 = arith.cmpf une, %right_841, %right_841 : tensor<8x128xf32>
    %cond_856 = arith.cmpf ogt, %left_840, %right_841 : tensor<8x128xf32>
    %cond_857 = arith.xori %right_isnan_855, %cst_0 : tensor<8x128xi1>
    %cond_858 = arith.andi %left_isnan_854, %cond_857 : tensor<8x128xi1>
    %cond_859 = arith.ori %cond_856, %cond_858 : tensor<8x128xi1>
    %eq_860 = arith.cmpf oeq, %left_840, %right_841 : tensor<8x128xf32>
    %eq_861 = arith.andi %left_isnan_854, %right_isnan_855 : tensor<8x128xi1>
    %eq_862 = arith.ori %eq_860, %eq_861 : tensor<8x128xi1>
    %cond_863 = arith.cmpi sgt, %left_idx_851, %right_idx_852 : tensor<8x128xi32>
    %cond_864 = arith.andi %eq_862, %cond_863 : tensor<8x128xi1>
    %cond_865 = arith.ori %cond_859, %cond_864 : tensor<8x128xi1>
    %cond_866 = arith.extui %cond_865 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_867 = arith.xori %cond_866, %flip_732 : tensor<8x128xi32>
    %cond_868 = arith.cmpi ne, %cond_867, %cst : tensor<8x128xi32>
    %ret_869 = arith.xori %ileft_838, %iright_839 : tensor<8x128xi32>
    %ret_870 = arith.select %cond_868, %ret_869, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_871 = arith.xori %ix_853, %ret_870 : tensor<8x128xi32>
    %new_idxs_872 = arith.xori %left_idx_851, %right_idx_852 : tensor<8x128xi32>
    %new_idxs_873 = arith.select %cond_868, %new_idxs_872, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_874 = arith.xori %new_idxs_827, %new_idxs_873 : tensor<8x128xi32>
    %17 = tt.bitcast %ret_871 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_875 = tt.reshape %17 : tensor<8x128xf32> -> tensor<128x2x4xf32>
    %iy_876 = tt.bitcast %y_875 : tensor<128x2x4xf32> -> tensor<128x2x4xi32>
    %ileft_877 = arith.muli %iy_876, %ileft_162 : tensor<128x2x4xi32>
    %ileft_878 = "tt.reduce"(%ileft_877) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %ileft_879 = tt.expand_dims %ileft_878 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %ileft_880 = tt.broadcast %ileft_879 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %iright_881 = arith.muli %iy_876, %flip_61 : tensor<128x2x4xi32>
    %iright_882 = "tt.reduce"(%iright_881) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %iright_883 = tt.expand_dims %iright_882 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %iright_884 = tt.broadcast %iright_883 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %ileft_885 = tt.reshape %ileft_880 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %iright_886 = tt.reshape %iright_884 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %left_887 = tt.bitcast %ileft_885 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_888 = tt.bitcast %iright_886 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_889 = tt.reshape %new_idxs_874 : tensor<8x128xi32> -> tensor<128x2x4xi32>
    %left_idx_890 = arith.muli %y_idx_889, %ileft_162 : tensor<128x2x4xi32>
    %left_idx_891 = "tt.reduce"(%left_idx_890) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %left_idx_892 = tt.expand_dims %left_idx_891 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %left_idx_893 = tt.broadcast %left_idx_892 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %right_idx_894 = arith.muli %y_idx_889, %flip_61 : tensor<128x2x4xi32>
    %right_idx_895 = "tt.reduce"(%right_idx_894) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %right_idx_896 = tt.expand_dims %right_idx_895 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %right_idx_897 = tt.broadcast %right_idx_896 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %left_idx_898 = tt.reshape %left_idx_893 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %right_idx_899 = tt.reshape %right_idx_897 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %ix_900 = tt.bitcast %17 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_901 = arith.cmpf une, %left_887, %left_887 : tensor<8x128xf32>
    %right_isnan_902 = arith.cmpf une, %right_888, %right_888 : tensor<8x128xf32>
    %cond_903 = arith.cmpf ogt, %left_887, %right_888 : tensor<8x128xf32>
    %cond_904 = arith.xori %right_isnan_902, %cst_0 : tensor<8x128xi1>
    %cond_905 = arith.andi %left_isnan_901, %cond_904 : tensor<8x128xi1>
    %cond_906 = arith.ori %cond_903, %cond_905 : tensor<8x128xi1>
    %eq_907 = arith.cmpf oeq, %left_887, %right_888 : tensor<8x128xf32>
    %eq_908 = arith.andi %left_isnan_901, %right_isnan_902 : tensor<8x128xi1>
    %eq_909 = arith.ori %eq_907, %eq_908 : tensor<8x128xi1>
    %cond_910 = arith.cmpi sgt, %left_idx_898, %right_idx_899 : tensor<8x128xi32>
    %cond_911 = arith.andi %eq_909, %cond_910 : tensor<8x128xi1>
    %cond_912 = arith.ori %cond_906, %cond_911 : tensor<8x128xi1>
    %cond_913 = arith.extui %cond_912 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_914 = arith.xori %cond_913, %flip_732 : tensor<8x128xi32>
    %cond_915 = arith.cmpi ne, %cond_914, %cst : tensor<8x128xi32>
    %ret_916 = arith.xori %ileft_885, %iright_886 : tensor<8x128xi32>
    %ret_917 = arith.select %cond_915, %ret_916, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_918 = arith.xori %ix_900, %ret_917 : tensor<8x128xi32>
    %new_idxs_919 = arith.xori %left_idx_898, %right_idx_899 : tensor<8x128xi32>
    %new_idxs_920 = arith.select %cond_915, %new_idxs_919, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_921 = arith.xori %new_idxs_874, %new_idxs_920 : tensor<8x128xi32>
    %18 = tt.bitcast %ret_918 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_922 = tt.reshape %18 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_923 = tt.bitcast %y_922 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_924 = arith.muli %iy_923, %ileft_65 : tensor<256x2x2xi32>
    %ileft_925 = "tt.reduce"(%ileft_924) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_926 = tt.expand_dims %ileft_925 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_927 = tt.broadcast %ileft_926 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_928 = arith.muli %iy_923, %flip_19 : tensor<256x2x2xi32>
    %iright_929 = "tt.reduce"(%iright_928) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_930 = tt.expand_dims %iright_929 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_931 = tt.broadcast %iright_930 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_932 = tt.reshape %ileft_927 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_933 = tt.reshape %iright_931 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_934 = tt.bitcast %ileft_932 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_935 = tt.bitcast %iright_933 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_936 = tt.reshape %new_idxs_921 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_937 = arith.muli %y_idx_936, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_938 = "tt.reduce"(%left_idx_937) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_939 = tt.expand_dims %left_idx_938 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_940 = tt.broadcast %left_idx_939 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_941 = arith.muli %y_idx_936, %flip_19 : tensor<256x2x2xi32>
    %right_idx_942 = "tt.reduce"(%right_idx_941) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_943 = tt.expand_dims %right_idx_942 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_944 = tt.broadcast %right_idx_943 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_945 = tt.reshape %left_idx_940 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_946 = tt.reshape %right_idx_944 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_947 = tt.bitcast %18 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_948 = arith.cmpf une, %left_934, %left_934 : tensor<8x128xf32>
    %right_isnan_949 = arith.cmpf une, %right_935, %right_935 : tensor<8x128xf32>
    %cond_950 = arith.cmpf ogt, %left_934, %right_935 : tensor<8x128xf32>
    %cond_951 = arith.xori %right_isnan_949, %cst_0 : tensor<8x128xi1>
    %cond_952 = arith.andi %left_isnan_948, %cond_951 : tensor<8x128xi1>
    %cond_953 = arith.ori %cond_950, %cond_952 : tensor<8x128xi1>
    %eq_954 = arith.cmpf oeq, %left_934, %right_935 : tensor<8x128xf32>
    %eq_955 = arith.andi %left_isnan_948, %right_isnan_949 : tensor<8x128xi1>
    %eq_956 = arith.ori %eq_954, %eq_955 : tensor<8x128xi1>
    %cond_957 = arith.cmpi sgt, %left_idx_945, %right_idx_946 : tensor<8x128xi32>
    %cond_958 = arith.andi %eq_956, %cond_957 : tensor<8x128xi1>
    %cond_959 = arith.ori %cond_953, %cond_958 : tensor<8x128xi1>
    %cond_960 = arith.extui %cond_959 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_961 = arith.xori %cond_960, %flip_732 : tensor<8x128xi32>
    %cond_962 = arith.cmpi ne, %cond_961, %cst : tensor<8x128xi32>
    %ret_963 = arith.xori %ileft_932, %iright_933 : tensor<8x128xi32>
    %ret_964 = arith.select %cond_962, %ret_963, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_965 = arith.xori %ix_947, %ret_964 : tensor<8x128xi32>
    %new_idxs_966 = arith.xori %left_idx_945, %right_idx_946 : tensor<8x128xi32>
    %new_idxs_967 = arith.select %cond_962, %new_idxs_966, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_968 = arith.xori %new_idxs_921, %new_idxs_967 : tensor<8x128xi32>
    %19 = tt.bitcast %ret_965 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_969 = tt.reshape %19 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_970 = tt.bitcast %y_969 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_971 = arith.muli %iy_970, %ileft : tensor<512x2x1xi32>
    %ileft_972 = "tt.reduce"(%ileft_971) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_973 = tt.expand_dims %ileft_972 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_974 = tt.broadcast %ileft_973 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_975 = arith.muli %iy_970, %iright : tensor<512x2x1xi32>
    %iright_976 = "tt.reduce"(%iright_975) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_977 = tt.expand_dims %iright_976 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_978 = tt.broadcast %iright_977 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_979 = tt.reshape %ileft_974 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_980 = tt.reshape %iright_978 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_981 = tt.bitcast %ileft_979 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_982 = tt.bitcast %iright_980 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_983 = tt.reshape %new_idxs_968 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_984 = arith.muli %y_idx_983, %ileft : tensor<512x2x1xi32>
    %left_idx_985 = "tt.reduce"(%left_idx_984) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_986 = tt.expand_dims %left_idx_985 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_987 = tt.broadcast %left_idx_986 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_988 = arith.muli %y_idx_983, %iright : tensor<512x2x1xi32>
    %right_idx_989 = "tt.reduce"(%right_idx_988) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_990 = tt.expand_dims %right_idx_989 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_991 = tt.broadcast %right_idx_990 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_992 = tt.reshape %left_idx_987 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_993 = tt.reshape %right_idx_991 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_994 = tt.bitcast %19 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_995 = arith.cmpf une, %left_981, %left_981 : tensor<8x128xf32>
    %right_isnan_996 = arith.cmpf une, %right_982, %right_982 : tensor<8x128xf32>
    %cond_997 = arith.cmpf ogt, %left_981, %right_982 : tensor<8x128xf32>
    %cond_998 = arith.xori %right_isnan_996, %cst_0 : tensor<8x128xi1>
    %cond_999 = arith.andi %left_isnan_995, %cond_998 : tensor<8x128xi1>
    %cond_1000 = arith.ori %cond_997, %cond_999 : tensor<8x128xi1>
    %eq_1001 = arith.cmpf oeq, %left_981, %right_982 : tensor<8x128xf32>
    %eq_1002 = arith.andi %left_isnan_995, %right_isnan_996 : tensor<8x128xi1>
    %eq_1003 = arith.ori %eq_1001, %eq_1002 : tensor<8x128xi1>
    %cond_1004 = arith.cmpi sgt, %left_idx_992, %right_idx_993 : tensor<8x128xi32>
    %cond_1005 = arith.andi %eq_1003, %cond_1004 : tensor<8x128xi1>
    %cond_1006 = arith.ori %cond_1000, %cond_1005 : tensor<8x128xi1>
    %cond_1007 = arith.extui %cond_1006 : tensor<8x128xi1> to tensor<8x128xi32>
    %cond_1008 = arith.xori %cond_1007, %flip_732 : tensor<8x128xi32>
    %cond_1009 = arith.cmpi ne, %cond_1008, %cst : tensor<8x128xi32>
    %ret_1010 = arith.xori %ileft_979, %iright_980 : tensor<8x128xi32>
    %ret_1011 = arith.select %cond_1009, %ret_1010, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1012 = arith.xori %ix_994, %ret_1011 : tensor<8x128xi32>
    %new_idxs_1013 = arith.xori %left_idx_992, %right_idx_993 : tensor<8x128xi32>
    %new_idxs_1014 = arith.select %cond_1009, %new_idxs_1013, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1015 = arith.xori %new_idxs_968, %new_idxs_1014 : tensor<8x128xi32>
    %20 = tt.bitcast %ret_1012 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1016 = tt.reshape %20 : tensor<8x128xf32> -> tensor<8x2x64xf32>
    %iy_1017 = tt.bitcast %y_1016 : tensor<8x2x64xf32> -> tensor<8x2x64xi32>
    %ileft_1018 = tt.broadcast %left_mask : tensor<1x2x1xi32> -> tensor<8x2x64xi32>
    %ileft_1019 = arith.muli %iy_1017, %ileft_1018 : tensor<8x2x64xi32>
    %ileft_1020 = "tt.reduce"(%ileft_1019) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<8x2x64xi32>) -> tensor<8x64xi32>
    %ileft_1021 = tt.expand_dims %ileft_1020 {axis = 1 : i32} : tensor<8x64xi32> -> tensor<8x1x64xi32>
    %ileft_1022 = tt.broadcast %ileft_1021 : tensor<8x1x64xi32> -> tensor<8x2x64xi32>
    %iright_1023 = arith.muli %iy_1017, %flip_731 : tensor<8x2x64xi32>
    %iright_1024 = "tt.reduce"(%iright_1023) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<8x2x64xi32>) -> tensor<8x64xi32>
    %iright_1025 = tt.expand_dims %iright_1024 {axis = 1 : i32} : tensor<8x64xi32> -> tensor<8x1x64xi32>
    %iright_1026 = tt.broadcast %iright_1025 : tensor<8x1x64xi32> -> tensor<8x2x64xi32>
    %ileft_1027 = tt.reshape %ileft_1022 : tensor<8x2x64xi32> -> tensor<8x128xi32>
    %iright_1028 = tt.reshape %iright_1026 : tensor<8x2x64xi32> -> tensor<8x128xi32>
    %left_1029 = tt.bitcast %ileft_1027 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1030 = tt.bitcast %iright_1028 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1031 = tt.reshape %new_idxs_1015 : tensor<8x128xi32> -> tensor<8x2x64xi32>
    %left_idx_1032 = arith.muli %y_idx_1031, %ileft_1018 : tensor<8x2x64xi32>
    %left_idx_1033 = "tt.reduce"(%left_idx_1032) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<8x2x64xi32>) -> tensor<8x64xi32>
    %left_idx_1034 = tt.expand_dims %left_idx_1033 {axis = 1 : i32} : tensor<8x64xi32> -> tensor<8x1x64xi32>
    %left_idx_1035 = tt.broadcast %left_idx_1034 : tensor<8x1x64xi32> -> tensor<8x2x64xi32>
    %right_idx_1036 = arith.muli %y_idx_1031, %flip_731 : tensor<8x2x64xi32>
    %right_idx_1037 = "tt.reduce"(%right_idx_1036) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<8x2x64xi32>) -> tensor<8x64xi32>
    %right_idx_1038 = tt.expand_dims %right_idx_1037 {axis = 1 : i32} : tensor<8x64xi32> -> tensor<8x1x64xi32>
    %right_idx_1039 = tt.broadcast %right_idx_1038 : tensor<8x1x64xi32> -> tensor<8x2x64xi32>
    %left_idx_1040 = tt.reshape %left_idx_1035 : tensor<8x2x64xi32> -> tensor<8x128xi32>
    %right_idx_1041 = tt.reshape %right_idx_1039 : tensor<8x2x64xi32> -> tensor<8x128xi32>
    %ix_1042 = tt.bitcast %20 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1043 = arith.cmpf une, %left_1029, %left_1029 : tensor<8x128xf32>
    %right_isnan_1044 = arith.cmpf une, %right_1030, %right_1030 : tensor<8x128xf32>
    %cond_1045 = arith.cmpf ogt, %left_1029, %right_1030 : tensor<8x128xf32>
    %cond_1046 = arith.xori %right_isnan_1044, %cst_0 : tensor<8x128xi1>
    %cond_1047 = arith.andi %left_isnan_1043, %cond_1046 : tensor<8x128xi1>
    %cond_1048 = arith.ori %cond_1045, %cond_1047 : tensor<8x128xi1>
    %eq_1049 = arith.cmpf oeq, %left_1029, %right_1030 : tensor<8x128xf32>
    %eq_1050 = arith.andi %left_isnan_1043, %right_isnan_1044 : tensor<8x128xi1>
    %eq_1051 = arith.ori %eq_1049, %eq_1050 : tensor<8x128xi1>
    %cond_1052 = arith.cmpi sgt, %left_idx_1040, %right_idx_1041 : tensor<8x128xi32>
    %cond_1053 = arith.andi %eq_1051, %cond_1052 : tensor<8x128xi1>
    %cond_1054 = arith.ori %cond_1048, %cond_1053 : tensor<8x128xi1>
    %ret_1055 = arith.xori %ileft_1027, %iright_1028 : tensor<8x128xi32>
    %ret_1056 = arith.select %cond_1054, %ret_1055, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1057 = arith.xori %ix_1042, %ret_1056 : tensor<8x128xi32>
    %new_idxs_1058 = arith.xori %left_idx_1040, %right_idx_1041 : tensor<8x128xi32>
    %new_idxs_1059 = arith.select %cond_1054, %new_idxs_1058, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1060 = arith.xori %new_idxs_1015, %new_idxs_1059 : tensor<8x128xi32>
    %21 = tt.bitcast %ret_1057 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1061 = tt.reshape %21 : tensor<8x128xf32> -> tensor<16x2x32xf32>
    %iy_1062 = tt.bitcast %y_1061 : tensor<16x2x32xf32> -> tensor<16x2x32xi32>
    %ileft_1063 = arith.muli %iy_1062, %ileft_735 : tensor<16x2x32xi32>
    %ileft_1064 = "tt.reduce"(%ileft_1063) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %ileft_1065 = tt.expand_dims %ileft_1064 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %ileft_1066 = tt.broadcast %ileft_1065 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %iright_1067 = arith.muli %iy_1062, %flip_493 : tensor<16x2x32xi32>
    %iright_1068 = "tt.reduce"(%iright_1067) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %iright_1069 = tt.expand_dims %iright_1068 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %iright_1070 = tt.broadcast %iright_1069 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %ileft_1071 = tt.reshape %ileft_1066 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %iright_1072 = tt.reshape %iright_1070 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %left_1073 = tt.bitcast %ileft_1071 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1074 = tt.bitcast %iright_1072 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1075 = tt.reshape %new_idxs_1060 : tensor<8x128xi32> -> tensor<16x2x32xi32>
    %left_idx_1076 = arith.muli %y_idx_1075, %ileft_735 : tensor<16x2x32xi32>
    %left_idx_1077 = "tt.reduce"(%left_idx_1076) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %left_idx_1078 = tt.expand_dims %left_idx_1077 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %left_idx_1079 = tt.broadcast %left_idx_1078 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %right_idx_1080 = arith.muli %y_idx_1075, %flip_493 : tensor<16x2x32xi32>
    %right_idx_1081 = "tt.reduce"(%right_idx_1080) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<16x2x32xi32>) -> tensor<16x32xi32>
    %right_idx_1082 = tt.expand_dims %right_idx_1081 {axis = 1 : i32} : tensor<16x32xi32> -> tensor<16x1x32xi32>
    %right_idx_1083 = tt.broadcast %right_idx_1082 : tensor<16x1x32xi32> -> tensor<16x2x32xi32>
    %left_idx_1084 = tt.reshape %left_idx_1079 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %right_idx_1085 = tt.reshape %right_idx_1083 : tensor<16x2x32xi32> -> tensor<8x128xi32>
    %ix_1086 = tt.bitcast %21 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1087 = arith.cmpf une, %left_1073, %left_1073 : tensor<8x128xf32>
    %right_isnan_1088 = arith.cmpf une, %right_1074, %right_1074 : tensor<8x128xf32>
    %cond_1089 = arith.cmpf ogt, %left_1073, %right_1074 : tensor<8x128xf32>
    %cond_1090 = arith.xori %right_isnan_1088, %cst_0 : tensor<8x128xi1>
    %cond_1091 = arith.andi %left_isnan_1087, %cond_1090 : tensor<8x128xi1>
    %cond_1092 = arith.ori %cond_1089, %cond_1091 : tensor<8x128xi1>
    %eq_1093 = arith.cmpf oeq, %left_1073, %right_1074 : tensor<8x128xf32>
    %eq_1094 = arith.andi %left_isnan_1087, %right_isnan_1088 : tensor<8x128xi1>
    %eq_1095 = arith.ori %eq_1093, %eq_1094 : tensor<8x128xi1>
    %cond_1096 = arith.cmpi sgt, %left_idx_1084, %right_idx_1085 : tensor<8x128xi32>
    %cond_1097 = arith.andi %eq_1095, %cond_1096 : tensor<8x128xi1>
    %cond_1098 = arith.ori %cond_1092, %cond_1097 : tensor<8x128xi1>
    %ret_1099 = arith.xori %ileft_1071, %iright_1072 : tensor<8x128xi32>
    %ret_1100 = arith.select %cond_1098, %ret_1099, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1101 = arith.xori %ix_1086, %ret_1100 : tensor<8x128xi32>
    %new_idxs_1102 = arith.xori %left_idx_1084, %right_idx_1085 : tensor<8x128xi32>
    %new_idxs_1103 = arith.select %cond_1098, %new_idxs_1102, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1104 = arith.xori %new_idxs_1060, %new_idxs_1103 : tensor<8x128xi32>
    %22 = tt.bitcast %ret_1101 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1105 = tt.reshape %22 : tensor<8x128xf32> -> tensor<32x2x16xf32>
    %iy_1106 = tt.bitcast %y_1105 : tensor<32x2x16xf32> -> tensor<32x2x16xi32>
    %ileft_1107 = arith.muli %iy_1106, %ileft_497 : tensor<32x2x16xi32>
    %ileft_1108 = "tt.reduce"(%ileft_1107) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %ileft_1109 = tt.expand_dims %ileft_1108 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %ileft_1110 = tt.broadcast %ileft_1109 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %iright_1111 = arith.muli %iy_1106, %flip_302 : tensor<32x2x16xi32>
    %iright_1112 = "tt.reduce"(%iright_1111) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %iright_1113 = tt.expand_dims %iright_1112 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %iright_1114 = tt.broadcast %iright_1113 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %ileft_1115 = tt.reshape %ileft_1110 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %iright_1116 = tt.reshape %iright_1114 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %left_1117 = tt.bitcast %ileft_1115 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1118 = tt.bitcast %iright_1116 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1119 = tt.reshape %new_idxs_1104 : tensor<8x128xi32> -> tensor<32x2x16xi32>
    %left_idx_1120 = arith.muli %y_idx_1119, %ileft_497 : tensor<32x2x16xi32>
    %left_idx_1121 = "tt.reduce"(%left_idx_1120) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %left_idx_1122 = tt.expand_dims %left_idx_1121 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %left_idx_1123 = tt.broadcast %left_idx_1122 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %right_idx_1124 = arith.muli %y_idx_1119, %flip_302 : tensor<32x2x16xi32>
    %right_idx_1125 = "tt.reduce"(%right_idx_1124) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<32x2x16xi32>) -> tensor<32x16xi32>
    %right_idx_1126 = tt.expand_dims %right_idx_1125 {axis = 1 : i32} : tensor<32x16xi32> -> tensor<32x1x16xi32>
    %right_idx_1127 = tt.broadcast %right_idx_1126 : tensor<32x1x16xi32> -> tensor<32x2x16xi32>
    %left_idx_1128 = tt.reshape %left_idx_1123 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %right_idx_1129 = tt.reshape %right_idx_1127 : tensor<32x2x16xi32> -> tensor<8x128xi32>
    %ix_1130 = tt.bitcast %22 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1131 = arith.cmpf une, %left_1117, %left_1117 : tensor<8x128xf32>
    %right_isnan_1132 = arith.cmpf une, %right_1118, %right_1118 : tensor<8x128xf32>
    %cond_1133 = arith.cmpf ogt, %left_1117, %right_1118 : tensor<8x128xf32>
    %cond_1134 = arith.xori %right_isnan_1132, %cst_0 : tensor<8x128xi1>
    %cond_1135 = arith.andi %left_isnan_1131, %cond_1134 : tensor<8x128xi1>
    %cond_1136 = arith.ori %cond_1133, %cond_1135 : tensor<8x128xi1>
    %eq_1137 = arith.cmpf oeq, %left_1117, %right_1118 : tensor<8x128xf32>
    %eq_1138 = arith.andi %left_isnan_1131, %right_isnan_1132 : tensor<8x128xi1>
    %eq_1139 = arith.ori %eq_1137, %eq_1138 : tensor<8x128xi1>
    %cond_1140 = arith.cmpi sgt, %left_idx_1128, %right_idx_1129 : tensor<8x128xi32>
    %cond_1141 = arith.andi %eq_1139, %cond_1140 : tensor<8x128xi1>
    %cond_1142 = arith.ori %cond_1136, %cond_1141 : tensor<8x128xi1>
    %ret_1143 = arith.xori %ileft_1115, %iright_1116 : tensor<8x128xi32>
    %ret_1144 = arith.select %cond_1142, %ret_1143, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1145 = arith.xori %ix_1130, %ret_1144 : tensor<8x128xi32>
    %new_idxs_1146 = arith.xori %left_idx_1128, %right_idx_1129 : tensor<8x128xi32>
    %new_idxs_1147 = arith.select %cond_1142, %new_idxs_1146, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1148 = arith.xori %new_idxs_1104, %new_idxs_1147 : tensor<8x128xi32>
    %23 = tt.bitcast %ret_1145 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1149 = tt.reshape %23 : tensor<8x128xf32> -> tensor<64x2x8xf32>
    %iy_1150 = tt.bitcast %y_1149 : tensor<64x2x8xf32> -> tensor<64x2x8xi32>
    %ileft_1151 = arith.muli %iy_1150, %ileft_306 : tensor<64x2x8xi32>
    %ileft_1152 = "tt.reduce"(%ileft_1151) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %ileft_1153 = tt.expand_dims %ileft_1152 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %ileft_1154 = tt.broadcast %ileft_1153 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %iright_1155 = arith.muli %iy_1150, %flip_158 : tensor<64x2x8xi32>
    %iright_1156 = "tt.reduce"(%iright_1155) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %iright_1157 = tt.expand_dims %iright_1156 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %iright_1158 = tt.broadcast %iright_1157 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %ileft_1159 = tt.reshape %ileft_1154 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %iright_1160 = tt.reshape %iright_1158 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %left_1161 = tt.bitcast %ileft_1159 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1162 = tt.bitcast %iright_1160 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1163 = tt.reshape %new_idxs_1148 : tensor<8x128xi32> -> tensor<64x2x8xi32>
    %left_idx_1164 = arith.muli %y_idx_1163, %ileft_306 : tensor<64x2x8xi32>
    %left_idx_1165 = "tt.reduce"(%left_idx_1164) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %left_idx_1166 = tt.expand_dims %left_idx_1165 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %left_idx_1167 = tt.broadcast %left_idx_1166 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %right_idx_1168 = arith.muli %y_idx_1163, %flip_158 : tensor<64x2x8xi32>
    %right_idx_1169 = "tt.reduce"(%right_idx_1168) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<64x2x8xi32>) -> tensor<64x8xi32>
    %right_idx_1170 = tt.expand_dims %right_idx_1169 {axis = 1 : i32} : tensor<64x8xi32> -> tensor<64x1x8xi32>
    %right_idx_1171 = tt.broadcast %right_idx_1170 : tensor<64x1x8xi32> -> tensor<64x2x8xi32>
    %left_idx_1172 = tt.reshape %left_idx_1167 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %right_idx_1173 = tt.reshape %right_idx_1171 : tensor<64x2x8xi32> -> tensor<8x128xi32>
    %ix_1174 = tt.bitcast %23 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1175 = arith.cmpf une, %left_1161, %left_1161 : tensor<8x128xf32>
    %right_isnan_1176 = arith.cmpf une, %right_1162, %right_1162 : tensor<8x128xf32>
    %cond_1177 = arith.cmpf ogt, %left_1161, %right_1162 : tensor<8x128xf32>
    %cond_1178 = arith.xori %right_isnan_1176, %cst_0 : tensor<8x128xi1>
    %cond_1179 = arith.andi %left_isnan_1175, %cond_1178 : tensor<8x128xi1>
    %cond_1180 = arith.ori %cond_1177, %cond_1179 : tensor<8x128xi1>
    %eq_1181 = arith.cmpf oeq, %left_1161, %right_1162 : tensor<8x128xf32>
    %eq_1182 = arith.andi %left_isnan_1175, %right_isnan_1176 : tensor<8x128xi1>
    %eq_1183 = arith.ori %eq_1181, %eq_1182 : tensor<8x128xi1>
    %cond_1184 = arith.cmpi sgt, %left_idx_1172, %right_idx_1173 : tensor<8x128xi32>
    %cond_1185 = arith.andi %eq_1183, %cond_1184 : tensor<8x128xi1>
    %cond_1186 = arith.ori %cond_1180, %cond_1185 : tensor<8x128xi1>
    %ret_1187 = arith.xori %ileft_1159, %iright_1160 : tensor<8x128xi32>
    %ret_1188 = arith.select %cond_1186, %ret_1187, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1189 = arith.xori %ix_1174, %ret_1188 : tensor<8x128xi32>
    %new_idxs_1190 = arith.xori %left_idx_1172, %right_idx_1173 : tensor<8x128xi32>
    %new_idxs_1191 = arith.select %cond_1186, %new_idxs_1190, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1192 = arith.xori %new_idxs_1148, %new_idxs_1191 : tensor<8x128xi32>
    %24 = tt.bitcast %ret_1189 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1193 = tt.reshape %24 : tensor<8x128xf32> -> tensor<128x2x4xf32>
    %iy_1194 = tt.bitcast %y_1193 : tensor<128x2x4xf32> -> tensor<128x2x4xi32>
    %ileft_1195 = arith.muli %iy_1194, %ileft_162 : tensor<128x2x4xi32>
    %ileft_1196 = "tt.reduce"(%ileft_1195) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %ileft_1197 = tt.expand_dims %ileft_1196 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %ileft_1198 = tt.broadcast %ileft_1197 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %iright_1199 = arith.muli %iy_1194, %flip_61 : tensor<128x2x4xi32>
    %iright_1200 = "tt.reduce"(%iright_1199) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %iright_1201 = tt.expand_dims %iright_1200 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %iright_1202 = tt.broadcast %iright_1201 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %ileft_1203 = tt.reshape %ileft_1198 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %iright_1204 = tt.reshape %iright_1202 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %left_1205 = tt.bitcast %ileft_1203 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1206 = tt.bitcast %iright_1204 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1207 = tt.reshape %new_idxs_1192 : tensor<8x128xi32> -> tensor<128x2x4xi32>
    %left_idx_1208 = arith.muli %y_idx_1207, %ileft_162 : tensor<128x2x4xi32>
    %left_idx_1209 = "tt.reduce"(%left_idx_1208) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %left_idx_1210 = tt.expand_dims %left_idx_1209 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %left_idx_1211 = tt.broadcast %left_idx_1210 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %right_idx_1212 = arith.muli %y_idx_1207, %flip_61 : tensor<128x2x4xi32>
    %right_idx_1213 = "tt.reduce"(%right_idx_1212) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<128x2x4xi32>) -> tensor<128x4xi32>
    %right_idx_1214 = tt.expand_dims %right_idx_1213 {axis = 1 : i32} : tensor<128x4xi32> -> tensor<128x1x4xi32>
    %right_idx_1215 = tt.broadcast %right_idx_1214 : tensor<128x1x4xi32> -> tensor<128x2x4xi32>
    %left_idx_1216 = tt.reshape %left_idx_1211 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %right_idx_1217 = tt.reshape %right_idx_1215 : tensor<128x2x4xi32> -> tensor<8x128xi32>
    %ix_1218 = tt.bitcast %24 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1219 = arith.cmpf une, %left_1205, %left_1205 : tensor<8x128xf32>
    %right_isnan_1220 = arith.cmpf une, %right_1206, %right_1206 : tensor<8x128xf32>
    %cond_1221 = arith.cmpf ogt, %left_1205, %right_1206 : tensor<8x128xf32>
    %cond_1222 = arith.xori %right_isnan_1220, %cst_0 : tensor<8x128xi1>
    %cond_1223 = arith.andi %left_isnan_1219, %cond_1222 : tensor<8x128xi1>
    %cond_1224 = arith.ori %cond_1221, %cond_1223 : tensor<8x128xi1>
    %eq_1225 = arith.cmpf oeq, %left_1205, %right_1206 : tensor<8x128xf32>
    %eq_1226 = arith.andi %left_isnan_1219, %right_isnan_1220 : tensor<8x128xi1>
    %eq_1227 = arith.ori %eq_1225, %eq_1226 : tensor<8x128xi1>
    %cond_1228 = arith.cmpi sgt, %left_idx_1216, %right_idx_1217 : tensor<8x128xi32>
    %cond_1229 = arith.andi %eq_1227, %cond_1228 : tensor<8x128xi1>
    %cond_1230 = arith.ori %cond_1224, %cond_1229 : tensor<8x128xi1>
    %ret_1231 = arith.xori %ileft_1203, %iright_1204 : tensor<8x128xi32>
    %ret_1232 = arith.select %cond_1230, %ret_1231, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1233 = arith.xori %ix_1218, %ret_1232 : tensor<8x128xi32>
    %new_idxs_1234 = arith.xori %left_idx_1216, %right_idx_1217 : tensor<8x128xi32>
    %new_idxs_1235 = arith.select %cond_1230, %new_idxs_1234, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1236 = arith.xori %new_idxs_1192, %new_idxs_1235 : tensor<8x128xi32>
    %25 = tt.bitcast %ret_1233 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1237 = tt.reshape %25 : tensor<8x128xf32> -> tensor<256x2x2xf32>
    %iy_1238 = tt.bitcast %y_1237 : tensor<256x2x2xf32> -> tensor<256x2x2xi32>
    %ileft_1239 = arith.muli %iy_1238, %ileft_65 : tensor<256x2x2xi32>
    %ileft_1240 = "tt.reduce"(%ileft_1239) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %ileft_1241 = tt.expand_dims %ileft_1240 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %ileft_1242 = tt.broadcast %ileft_1241 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %iright_1243 = arith.muli %iy_1238, %flip_19 : tensor<256x2x2xi32>
    %iright_1244 = "tt.reduce"(%iright_1243) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %iright_1245 = tt.expand_dims %iright_1244 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %iright_1246 = tt.broadcast %iright_1245 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %ileft_1247 = tt.reshape %ileft_1242 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %iright_1248 = tt.reshape %iright_1246 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %left_1249 = tt.bitcast %ileft_1247 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1250 = tt.bitcast %iright_1248 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1251 = tt.reshape %new_idxs_1236 : tensor<8x128xi32> -> tensor<256x2x2xi32>
    %left_idx_1252 = arith.muli %y_idx_1251, %ileft_65 : tensor<256x2x2xi32>
    %left_idx_1253 = "tt.reduce"(%left_idx_1252) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %left_idx_1254 = tt.expand_dims %left_idx_1253 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %left_idx_1255 = tt.broadcast %left_idx_1254 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %right_idx_1256 = arith.muli %y_idx_1251, %flip_19 : tensor<256x2x2xi32>
    %right_idx_1257 = "tt.reduce"(%right_idx_1256) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<256x2x2xi32>) -> tensor<256x2xi32>
    %right_idx_1258 = tt.expand_dims %right_idx_1257 {axis = 1 : i32} : tensor<256x2xi32> -> tensor<256x1x2xi32>
    %right_idx_1259 = tt.broadcast %right_idx_1258 : tensor<256x1x2xi32> -> tensor<256x2x2xi32>
    %left_idx_1260 = tt.reshape %left_idx_1255 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %right_idx_1261 = tt.reshape %right_idx_1259 : tensor<256x2x2xi32> -> tensor<8x128xi32>
    %ix_1262 = tt.bitcast %25 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1263 = arith.cmpf une, %left_1249, %left_1249 : tensor<8x128xf32>
    %right_isnan_1264 = arith.cmpf une, %right_1250, %right_1250 : tensor<8x128xf32>
    %cond_1265 = arith.cmpf ogt, %left_1249, %right_1250 : tensor<8x128xf32>
    %cond_1266 = arith.xori %right_isnan_1264, %cst_0 : tensor<8x128xi1>
    %cond_1267 = arith.andi %left_isnan_1263, %cond_1266 : tensor<8x128xi1>
    %cond_1268 = arith.ori %cond_1265, %cond_1267 : tensor<8x128xi1>
    %eq_1269 = arith.cmpf oeq, %left_1249, %right_1250 : tensor<8x128xf32>
    %eq_1270 = arith.andi %left_isnan_1263, %right_isnan_1264 : tensor<8x128xi1>
    %eq_1271 = arith.ori %eq_1269, %eq_1270 : tensor<8x128xi1>
    %cond_1272 = arith.cmpi sgt, %left_idx_1260, %right_idx_1261 : tensor<8x128xi32>
    %cond_1273 = arith.andi %eq_1271, %cond_1272 : tensor<8x128xi1>
    %cond_1274 = arith.ori %cond_1268, %cond_1273 : tensor<8x128xi1>
    %ret_1275 = arith.xori %ileft_1247, %iright_1248 : tensor<8x128xi32>
    %ret_1276 = arith.select %cond_1274, %ret_1275, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1277 = arith.xori %ix_1262, %ret_1276 : tensor<8x128xi32>
    %new_idxs_1278 = arith.xori %left_idx_1260, %right_idx_1261 : tensor<8x128xi32>
    %new_idxs_1279 = arith.select %cond_1274, %new_idxs_1278, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1280 = arith.xori %new_idxs_1236, %new_idxs_1279 : tensor<8x128xi32>
    %26 = tt.bitcast %ret_1277 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_1281 = tt.reshape %26 : tensor<8x128xf32> -> tensor<512x2x1xf32>
    %iy_1282 = tt.bitcast %y_1281 : tensor<512x2x1xf32> -> tensor<512x2x1xi32>
    %ileft_1283 = arith.muli %iy_1282, %ileft : tensor<512x2x1xi32>
    %ileft_1284 = "tt.reduce"(%ileft_1283) <{axis = 1 : i32}> ({
    ^bb0(%ileft_1325: i32, %ileft_1326: i32):
      %ileft_1327 = arith.addi %ileft_1325, %ileft_1326 : i32
      tt.reduce.return %ileft_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %ileft_1285 = tt.expand_dims %ileft_1284 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %ileft_1286 = tt.broadcast %ileft_1285 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %iright_1287 = arith.muli %iy_1282, %iright : tensor<512x2x1xi32>
    %iright_1288 = "tt.reduce"(%iright_1287) <{axis = 1 : i32}> ({
    ^bb0(%iright_1325: i32, %iright_1326: i32):
      %iright_1327 = arith.addi %iright_1325, %iright_1326 : i32
      tt.reduce.return %iright_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %iright_1289 = tt.expand_dims %iright_1288 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %iright_1290 = tt.broadcast %iright_1289 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %ileft_1291 = tt.reshape %ileft_1286 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %iright_1292 = tt.reshape %iright_1290 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %left_1293 = tt.bitcast %ileft_1291 : tensor<8x128xi32> -> tensor<8x128xf32>
    %right_1294 = tt.bitcast %iright_1292 : tensor<8x128xi32> -> tensor<8x128xf32>
    %y_idx_1295 = tt.reshape %new_idxs_1280 : tensor<8x128xi32> -> tensor<512x2x1xi32>
    %left_idx_1296 = arith.muli %y_idx_1295, %ileft : tensor<512x2x1xi32>
    %left_idx_1297 = "tt.reduce"(%left_idx_1296) <{axis = 1 : i32}> ({
    ^bb0(%left_idx_1325: i32, %left_idx_1326: i32):
      %left_idx_1327 = arith.addi %left_idx_1325, %left_idx_1326 : i32
      tt.reduce.return %left_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %left_idx_1298 = tt.expand_dims %left_idx_1297 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %left_idx_1299 = tt.broadcast %left_idx_1298 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %right_idx_1300 = arith.muli %y_idx_1295, %iright : tensor<512x2x1xi32>
    %right_idx_1301 = "tt.reduce"(%right_idx_1300) <{axis = 1 : i32}> ({
    ^bb0(%right_idx_1325: i32, %right_idx_1326: i32):
      %right_idx_1327 = arith.addi %right_idx_1325, %right_idx_1326 : i32
      tt.reduce.return %right_idx_1327 : i32
    }) : (tensor<512x2x1xi32>) -> tensor<512x1xi32>
    %right_idx_1302 = tt.expand_dims %right_idx_1301 {axis = 1 : i32} : tensor<512x1xi32> -> tensor<512x1x1xi32>
    %right_idx_1303 = tt.broadcast %right_idx_1302 : tensor<512x1x1xi32> -> tensor<512x2x1xi32>
    %left_idx_1304 = tt.reshape %left_idx_1299 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %right_idx_1305 = tt.reshape %right_idx_1303 : tensor<512x2x1xi32> -> tensor<8x128xi32>
    %ix_1306 = tt.bitcast %26 : tensor<8x128xf32> -> tensor<8x128xi32>
    %left_isnan_1307 = arith.cmpf une, %left_1293, %left_1293 : tensor<8x128xf32>
    %right_isnan_1308 = arith.cmpf une, %right_1294, %right_1294 : tensor<8x128xf32>
    %cond_1309 = arith.cmpf ogt, %left_1293, %right_1294 : tensor<8x128xf32>
    %cond_1310 = arith.xori %right_isnan_1308, %cst_0 : tensor<8x128xi1>
    %cond_1311 = arith.andi %left_isnan_1307, %cond_1310 : tensor<8x128xi1>
    %cond_1312 = arith.ori %cond_1309, %cond_1311 : tensor<8x128xi1>
    %eq_1313 = arith.cmpf oeq, %left_1293, %right_1294 : tensor<8x128xf32>
    %eq_1314 = arith.andi %left_isnan_1307, %right_isnan_1308 : tensor<8x128xi1>
    %eq_1315 = arith.ori %eq_1313, %eq_1314 : tensor<8x128xi1>
    %cond_1316 = arith.cmpi sgt, %left_idx_1304, %right_idx_1305 : tensor<8x128xi32>
    %cond_1317 = arith.andi %eq_1315, %cond_1316 : tensor<8x128xi1>
    %cond_1318 = arith.ori %cond_1312, %cond_1317 : tensor<8x128xi1>
    %ret_1319 = arith.xori %ileft_1291, %iright_1292 : tensor<8x128xi32>
    %ret_1320 = arith.select %cond_1318, %ret_1319, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %ret_1321 = arith.xori %ix_1306, %ret_1320 : tensor<8x128xi32>
    %new_idxs_1322 = arith.xori %left_idx_1304, %right_idx_1305 : tensor<8x128xi32>
    %new_idxs_1323 = arith.select %cond_1318, %new_idxs_1322, %cst : tensor<8x128xi1>, tensor<8x128xi32>
    %new_idxs_1324 = arith.xori %new_idxs_1280, %new_idxs_1323 : tensor<8x128xi32>
    %27 = tt.bitcast %ret_1321 : tensor<8x128xi32> -> tensor<8x128xf32>
    %tmp7 = arith.extsi %new_idxs_1324 : tensor<8x128xi32> to tensor<8x128xi64>
    %28 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<8x128x!tt.ptr<f32>>
    %29 = tt.addptr %28, %tmp0_12 : tensor<8x128x!tt.ptr<f32>>, tensor<8x128xi32>
    tt.store %29, %27, %tmp0_15 : tensor<8x128x!tt.ptr<f32>>
    %30 = tt.splat %out_ptr2 : !tt.ptr<i64> -> tensor<8x128x!tt.ptr<i64>>
    %31 = tt.addptr %30, %tmp0_12 : tensor<8x128x!tt.ptr<i64>>, tensor<8x128xi32>
    tt.store %31, %tmp7, %tmp0_15 : tensor<8x128x!tt.ptr<i64>>
    tt.return
  }
}
    """

    temp_file = tmp_path / "test_regression.ttir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file), options={"num_ctas": 1, "num_stages": 1, "num_warps": 2})

    from triton.runtime.driver import driver
    device = driver.active.get_current_device()

    # try to catch:
    # L0 build module failed. Log: IGC: Internal Compiler Error: Segmentation violation
    # Error during Intel loadBinary: Triton Error [ZE]: 0x70000004
    # RuntimeError: Triton Error [ZE]: 0x70000004
    module, function, n_regs, n_spills, n_max_threads = driver.active.utils.load_binary(
        kernel.name, kernel.kernel, kernel.metadata.shared, kernel.metadata.build_flags,
        not kernel.metadata.generate_native_code, device)
