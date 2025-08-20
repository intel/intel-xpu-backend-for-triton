// RUN: TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=ONE-MATRIX-CHECK
// RUN: TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=0 triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=STD-CHECK,CHECK


// COM: A matrix, 16x16 block size, 1 warp w/ repCluster=1
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 block size, 1 warp w/ repCluster=2
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 block size, 1 warp w/ repCluster=4
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 16x16 block size, 1 warp w/ repCluster=8
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 16 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 block size, 1 warp w/ repCluster=1
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 block size, 1 warp w/ repCluster=2
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 block size, 1 warp w/ repCluster=4
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32x16 block size, 1 warp w/ repCluster=8
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 32 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<32x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 block size, 1 warp w/ repCluster=1
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 block size, 1 warp w/ repCluster=2
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x16 block size, 1 warp w/ repCluster=4
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 block size, 1 warp w/ repCluster=1
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 block size, 1 warp w/ repCluster=2
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 64x32 block size, 1 warp w/ repCluster=4
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 128x32 block size, 1 warp w/ repCluster=8 (capped load height at 32)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [8, 1]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 128 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix, 32 warps, 256x32 block size (from AxB benchmark)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 256 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-1: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: B matrix, 32 warps, 32x256 block size (from AxB benchmark)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 256 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %4 = tt.make_tensor_ptr %arg2, [%N_i64, %M_i64], [%N_i64, %c1_i64], [%c0_i32, %0] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %5 = tt.load %4 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major" } : !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix, 32 warps, 32x256 block size, transpose (from AxBT benchmark)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 256 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %4 = tt.make_tensor_ptr %arg2, [%N_i64, %M_i64], [%c1_i64, %N_i64], [%c0_i32, %0] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %5 = tt.load %4 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major" } : !tt.ptr<tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: A matrix with 16 warps, 128x128 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 1], repCluster = [1, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 128 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix with 8 warps, 128x64 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 128 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: A matrix with 8 warps, 128x128 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 128 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 16 warps, 64x64 block size, transpose (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // STD-CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // ONE-MATRIX-CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%c1_i64, %N_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 16 warps, 128x64 block size, transpose (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 1], repCluster = [1, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // STD-CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // ONE-MATRIX-CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%c1_i64, %N_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 8 warps, 128x64 block size, transpose (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // STD-CHECK-COUNT-16: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 32, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // ONE-MATRIX-CHECK-COUNT-32: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 16, v_blocks = 1, transpose = true, vnni_transform = false, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%c1_i64, %N_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "column_major"} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 8 warps, 64x64 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 64 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-4: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 8 warps, 64x128 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [2, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}

// -----

// COM: B matrix with 16 warps, 64x128 block size (from flex attention)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 1], repCluster = [1, 2]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
    tt.func public @subgroup_2d_block_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16: i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16: i32}) attributes {noinline = false} {
        %0 = tt.get_program_id x : i32
        %M_i64 = arith.constant 64 : i64
        %N_i64 = arith.constant 128 : i64
        %c1_i64 = arith.constant 1 : i64
        %c0_i32 = arith.constant 0 : i32

        // CHECK-COUNT-8: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 32, v_blocks = 2, transpose = false, vnni_transform = true, cache_control = Default}
        // CHECK-NOT: triton_gen.2Dblockload
        %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
        %2 = tt.load %1 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>

        tt.return
    }
}
