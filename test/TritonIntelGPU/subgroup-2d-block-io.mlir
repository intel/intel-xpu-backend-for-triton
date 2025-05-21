// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s 
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm=one_matrix_per_load_for_bt=1 | FileCheck %s


// COM: A matrix, 16x16 w/ repCluster=1 --> 2x 8x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt 
        // CHECK-NOT: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 w/ repCluster=2 --> 1x 16x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt 
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 w/ repCluster=4 --> 1x 32x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt 
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 w/ repCluster=8 --> 2x 32x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt 
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 w/ repCluster=1 --> 4x 8x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 w/ repCluster=2 --> 2x 16x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 w/ repCluster=4 --> 1x 16x32x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 w/ repCluster=8 --> 2x 16x32x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 w/ repCluster=1 --> 8x 8x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 w/ repCluster=2 --> 4x 16x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_16r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 w/ repCluster=4 --> 2x 32x16x1 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 16 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x1cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 w/ repCluster=1 --> 8x 8x16x2 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 32 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x2cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 w/ repCluster=2 --> 4x 16x16x2 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 32 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i16
        // CHECK-NOT: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i16
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 w/ repCluster=4 --> 2x 16x32x2 loads 
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 32 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 128x32 w/ repCluster=8 --> 4x 16x32x2 loads  (capped load height at 32)
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block, triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 128 : i64
        %N_i64 = arith.constant 32 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt
        // CHECK-NOT: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, triton_intel_gpu.block_io = "row_major"} : !tt.ptr<tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}