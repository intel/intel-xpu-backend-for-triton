// RUN: triton-opt %s -tritonintelgpu-prefetch-block=inject-split-barriers=true | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], warpsPerCTA = [8, 4], order = [1, 0]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

// CHECK-DAG: #blocked1 = #ttg.blocked<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], warpsPerCTA = [32, 1], order = [1, 0]}>
// CHECK-DAG: #blocked2 = #ttg.blocked<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], warpsPerCTA = [4, 8], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16, 1>, %arg1: !tt.ptr<f16, 1>, %arg2: !tt.ptr<f32, 1>) {
    // CHECK-LABEL: @matmul_kernel_with_block_pointers
    // CHECK-DAG:  [[CST_ZERO:%.*]] = arith.constant 0 : i32
    // CHECK-DAG:  [[CST_32:%.*]] = arith.constant 32 : i32
    // CHECK-DAG:  [[CST_4096:%.*]] = arith.constant 4096 : i32

    // COM: Prefetch the 1st operand of the `tl.dot` operation 3 iterations in advance
    // CHECK:      [[A0:%.*]] = tt.make_tensor_ptr %arg0, {{.*}} : <tensor<256x32xf16, #blocked1>>
    // CHECK-NEXT: ttig.prefetch [[A0]] {{.*}} : !tt.ptr<tensor<256x32xf16, #blocked1>>
    // CHECK:      [[A1:%.*]] = tt.advance [[A0]], {{.*}} : <tensor<256x32xf16, #blocked1>>
    // CHECK-NEXT: ttig.prefetch [[A1]] {{.*}} : !tt.ptr<tensor<256x32xf16, #blocked1>>
    // CHECK:      [[A2:%.*]] = tt.advance [[A1]], {{.*}} : <tensor<256x32xf16, #blocked1>>
    // CHECK-NEXT: ttig.prefetch [[A2]] {{.*}} : !tt.ptr<tensor<256x32xf16, #blocked1>>
    // CHECK-NEXT: [[A3:%.*]] = tt.advance [[A2]], {{.*}} : <tensor<256x32xf16, #blocked1>>
    // CHECK-NEXT: [[A4:%.*]] = tt.make_tensor_ptr %arg0, {{.*}} : <tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>>

    // COM: Prefetch the 2nd operand of the `tl.dot` operation 3 iterations in advance
    // CHECK:      [[B0:%.*]] = tt.make_tensor_ptr %arg1, {{.*}} : <tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT: ttig.prefetch [[B0]] {{.*}} : !tt.ptr<tensor<32x256xf16, #blocked2>>
    // CHECK:      [[B1:%.*]] = tt.advance [[B0]], {{.*}} : <tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT: ttig.prefetch [[B1]] {{.*}} : !tt.ptr<tensor<32x256xf16, #blocked2>>
    // CHECK:      [[B2:%.*]] = tt.advance [[B1]], {{.*}} : <tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT: ttig.prefetch [[B2]] {{.*}} : !tt.ptr<tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT: [[B3:%.*]] = tt.advance [[B2]], {{.*}} : <tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT: [[B4:%.*]] = tt.make_tensor_ptr %arg1, {{.*}} : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>>

    // CHECK:      spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <None>
    // CHECK-NEXT: scf.for [[IV:%.*]] = [[CST_ZERO]] to [[CST_4096]] step [[CST_32]]
    // CHECK-SAME:      iter_args([[CST:%.*]] = {{.*}}, [[A6:%.*]] = [[A4]], [[B6:%.*]] = [[B4]], [[A5:%.*]] = [[A3]], [[B5:%.*]] = [[B3]])
    // CHECK-NEXT:   [[LD_A:%.*]] = tt.load [[A6]]
    // CHECK-NEXT:   [[LD_B:%.*]] = tt.load [[B6]]
    // CHECK:        ttig.prefetch [[A5]] {{.*}} : !tt.ptr<tensor<256x32xf16, #blocked1>>
    // CHECK:        ttig.prefetch [[B5]] {{.*}} : !tt.ptr<tensor<32x256xf16, #blocked2>>
    // CHECK-NEXT:   [[DOT:%.*]] = tt.dot [[LD_A]], [[LD_B]], [[CST]]
    // CHECK-NEXT:   tt.advance [[A5]], {{.*}} : <tensor<256x32xf16, #blocked1>>
    // CHECK-DAG:    tt.advance [[A6]], {{.*}} : <tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>>
    // CHECK-NEXT:   tt.advance [[B5]], {{.*}} : <tensor<32x256xf16, #blocked2>>
    // CHECK-DAG:    tt.advance [[B6]], {{.*}} : <tensor<32x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>>
    // CHECK:        spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <None>
    // CHECK-NEXT:   spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <None>
    // CHECK-NEXT:   scf.yield {{.*}}
    // CHECK-NEXT: }
    // CHECK-NEXT: spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <None>

    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c16_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %4 : i32
    %6 = arith.addi %2, %5 : i32
    %7 = arith.remsi %0, %c64_i32 : i32
    %8 = arith.divsi %7, %4 : i32
    %9 = arith.muli %6, %c256_i32 : i32
    %10 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #dot0>, 1>
    %11 = arith.muli %8, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %11] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #dot1>, 1>
    %13:3 = scf.for %arg6 = %c0_i32 to %c4096_i32 step %c32_i32 iter_args(%arg7 = %cst, %arg8 = %10, %arg9 = %12) -> (tensor<256x256xf32, #blocked>, !tt.ptr<tensor<256x32xf16, #dot0>, 1>, !tt.ptr<tensor<32x256xf16, #dot1>, 1>) : i32 {
      %15 = tt.load %arg8 : !tt.ptr<tensor<256x32xf16, #dot0>, 1>
      %16 = tt.load %arg9 : !tt.ptr<tensor<32x256xf16, #dot1>, 1>
      %17 = tt.dot %15, %16, %arg7 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #blocked>
      %18 = tt.advance %arg8, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #dot0>, 1>
      %19 = tt.advance %arg9, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #dot1>, 1>
      scf.yield %17, %18, %19 : tensor<256x256xf32, #blocked>, !tt.ptr<tensor<256x32xf16, #dot0>, 1>, !tt.ptr<tensor<32x256xf16, #dot1>, 1>
    } {ttg.workload = 3 : i32}
    %14 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%9, %11] {order = array<i32: 1, 0>} : <tensor<256x256xf32, #blocked>, 1>
    tt.store %14, %13#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<256x256xf32, #blocked>, 1>
    tt.return
  }
}
