// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory  --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_kernelcc @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK-SAME: attributes {triton_gen.intel_reqd_sub_group_size = [32 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {
    // CHECK:  llvm.return
    tt.return
  }
} // end module

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  tt.func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1)>
    // CHECK-NEXT: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1)>
    // CHECK-NEXT: [[ARG2_0:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(f32, f32)>
    // CHECK:      [[VEC:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i64] : vector<1xf32>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i32)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i32 to vector<1xf32>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf32>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i32)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i32 to vector<1xf32>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf32>
    // CHECK-NEXT:   [[RES1:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES2:%.*]] = llvm.insertvalue [[EE1]], [[RES1]][0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES3:%.*]] = llvm.insertvalue [[EE2]], [[RES2]][1] : !llvm.struct<(f32, f32)>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load
  tt.func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1)>
    // CHECK-NEXT: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1)>
    // CHECK-NEXT: [[ARG2_0:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(f32, f32)>
    // CHECK:      [[VEC:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i64] : vector<1xf32>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i32)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i32 to vector<1xf32>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf32>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i32)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i32 to vector<1xf32>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf32>
    // CHECK-NEXT:   [[RES1:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES2:%.*]] = llvm.insertvalue [[EE1]], [[RES1]][0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES3:%.*]] = llvm.insertvalue [[EE2]], [[RES2]][1] : !llvm.struct<(f32, f32)>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: vectorized_load_f16
  tt.func @vectorized_load_f16(%a_ptr_init: tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_2:%.*]] = llvm.extractvalue %arg0[2] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_3:%.*]] = llvm.extractvalue %arg0[3] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_4:%.*]] = llvm.extractvalue %arg0[4] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_5:%.*]] = llvm.extractvalue %arg0[5] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_6:%.*]] = llvm.extractvalue %arg0[6] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG0_7:%.*]] = llvm.extractvalue %arg0[7] : !llvm.struct<(ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>, ptr<1>)>
    // CHECK:      [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_2:%.*]] = llvm.extractvalue %arg1[2] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_3:%.*]] = llvm.extractvalue %arg1[3] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_4:%.*]] = llvm.extractvalue %arg1[4] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_5:%.*]] = llvm.extractvalue %arg1[5] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_6:%.*]] = llvm.extractvalue %arg1[6] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG1_7:%.*]] = llvm.extractvalue %arg1[7] : !llvm.struct<(i1, i1, i1, i1, i1, i1, i1, i1)>
    // CHECK-NEXT: [[ARG2_0:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK:      [[VEC:%.*]] = llvm.mlir.undef : vector<1xf16>
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i64
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i64] : vector<1xf16>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast [[IE1]] : vector<1xf16> to i16
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i16)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i16)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i16 to vector<1xf16>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i16)

    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i16)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i16 to vector<1xf16>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST4:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_2]], ^bb5, ^bb6([[BCAST4]] : i16)

    // CHECK-NEXT: ^bb5:
    // CHECK-NEXT:   [[BCAST5:%.*]] = llvm.bitcast [[ARG0_2]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD3:%.*]] = llvm.load [[BCAST5]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb6([[LOAD3]] : i16)
    // CHECK-NEXT: ^bb6([[V3:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V3:%.*]] = llvm.bitcast [[V3]] : i16 to vector<1xf16>
    // CHECK:        [[EE3:%.*]] = llvm.extractelement [[BCAST_V3]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST5:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_3]], ^bb7, ^bb8([[BCAST5]] : i16)

    // CHECK-NEXT: ^bb7:
    // CHECK:        [[BCAST6:%.*]] = llvm.bitcast [[ARG0_3]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD4:%.*]] = llvm.load [[BCAST6]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb8([[LOAD4]] : i16)
    // CHECK-NEXT: ^bb8([[V4:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V4:%.*]] = llvm.bitcast [[V4]] : i16 to vector<1xf16>
    // CHECK:        [[EE4:%.*]] = llvm.extractelement [[BCAST_V4]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST7:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_4]], ^bb9, ^bb10([[BCAST7]] : i16)

    // CHECK-NEXT: ^bb9:
    // CHECK:        [[BCAST8:%.*]] = llvm.bitcast [[ARG0_4]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD5:%.*]] = llvm.load [[BCAST8]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb10([[LOAD5]] : i16)
    // CHECK-NEXT: ^bb10([[V5:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V5:%.*]] = llvm.bitcast [[V5]] : i16 to vector<1xf16>
    // CHECK:        [[EE5:%.*]] = llvm.extractelement [[BCAST_V5]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST8:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_5]], ^bb11, ^bb12([[BCAST8]] : i16)

    // CHECK-NEXT: ^bb11:
    // CHECK:        [[BCAST9:%.*]] = llvm.bitcast [[ARG0_5]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD6:%.*]] = llvm.load [[BCAST9]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb12([[LOAD6]] : i16)
    // CHECK-NEXT: ^bb12([[V6:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V6:%.*]] = llvm.bitcast [[V6]] : i16 to vector<1xf16>
    // CHECK:        [[EE6:%.*]] = llvm.extractelement [[BCAST_V6]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST10:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_6]], ^bb13, ^bb14([[BCAST10]] : i16)

    // CHECK-NEXT: ^bb13:
    // CHECK:        [[BCAST11:%.*]] = llvm.bitcast [[ARG0_6]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD7:%.*]] = llvm.load [[BCAST11]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb14([[LOAD7]] : i16)
    // CHECK-NEXT: ^bb14([[V7:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V7:%.*]] = llvm.bitcast [[V7]] : i16 to vector<1xf16>
    // CHECK:        [[EE7:%.*]] = llvm.extractelement [[BCAST_V7]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST12:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_7]], ^bb15, ^bb16([[BCAST12]] : i16)

    // CHECK-NEXT: ^bb15:
    // CHECK:        [[BCAST13:%.*]] = llvm.bitcast [[ARG0_7]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD8:%.*]] = llvm.load [[BCAST13]] {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb16([[LOAD8]] : i16)
    // CHECK-NEXT: ^bb16([[V8:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V8:%.*]] = llvm.bitcast [[V8]] : i16 to vector<1xf16>
    // CHECK:        [[EE8:%.*]] = llvm.extractelement [[BCAST_V8]][{{.*}} : i32] : vector<1xf16>
    // CHECK-NEXT:   [[RES1:%.*]] = llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES2:%.*]] = llvm.insertvalue [[EE1]], [[RES1]][0] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES3:%.*]] = llvm.insertvalue [[EE2]], [[RES2]][1] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES4:%.*]] = llvm.insertvalue [[EE3]], [[RES3]][2] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES5:%.*]] = llvm.insertvalue [[EE4]], [[RES4]][3] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES6:%.*]] = llvm.insertvalue [[EE5]], [[RES5]][4] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES7:%.*]] = llvm.insertvalue [[EE6]], [[RES6]][5] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES8:%.*]] = llvm.insertvalue [[EE7]], [[RES7]][6] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    // CHECK-NEXT:   [[RES9:%.*]] = llvm.insertvalue [[EE8]], [[RES8]][7] : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other
  tt.func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// TODO: masked load with vectorization is pending on TODO
#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: masked_load_const_other_vec
  tt.func @masked_load_const_other_vec(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_with_cache_attr
  tt.func @store_with_cache_attr(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    tt.store %a_ptr_init, %cst_0, %cst evictionPolicy = evict_last cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_no_vec
  tt.func @global_load_store_no_vec(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from vector0
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32

    // Load 4 elements from vector1
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_vec4
  tt.func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// This test verifies the vectorization of Load and Store Ops.
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
// Note, the %n_elements doesn't have a "tt.divisibility" hint, so Triton assumes it's divisibility is 1, this should effect the mask's alignment and further restrict the load/store ops' vector width to be 1.
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  tt.func @vecadd_masked_vec1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<64xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<64xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %9 = tt.splat %n_elements : i32 -> tensor<64xi32, #blocked>
    %10 = arith.cmpi "slt", %4, %9 : tensor<64xi32, #blocked>
    // load op has a vector width = 1 due to the %mask's alignment
    // CHECK: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    %11 = tt.load %6, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %12 = tt.load %8, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    %13 = arith.addf %11, %12 : tensor<64xf32, #blocked>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    tt.store %15, %13, %10 : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>


    // Load 8 elements from B with four vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
    // CHECK-LABEL: global_load_store_vec2
    tt.func @global_load_store_vec2(%arg0: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 8 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with four vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>

    // Load 8 elements from B with four vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> vector<2xi32>

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with four vectorized store instruction
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 8 : i64} : vector<2xi32>, !llvm.ptr<1>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_store_vec8
    tt.func @global_load_store_vec8(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Load 8 elements from A with two vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>

    // Load 8 elements from B with two vectorized load instruction
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK: llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>

    %9 = tt.load %6 : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %8 : tensor<256x!tt.ptr<f32>, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>

    // Store 8 elements to global with two vectorized store instruction
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alignment = 16 : i64} : vector<4xi32>, !llvm.ptr<1>
    tt.store %13, %11 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// TODO: Add a testcase to verify the optimization when ptr of the LoadOp
//       is from an addptr with const idx

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_view_broadcast
  tt.func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0]
    // CHECK-NEXT: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1]
    // CHECK-NEXT: [[STRUCT:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[STRUCT1:%.*]] = llvm.insertvalue [[ARG0_0]], [[STRUCT]][0]
    // CHECK-NEXT: [[STRUCT2:%.*]] = llvm.insertvalue [[ARG0_1]], [[STRUCT1]][1]
    // CHECK-NEXT: [[T0:%.*]] = llvm.extractvalue [[STRUCT2]][0]
    // CHECK-NEXT: [[T1:%.*]] = llvm.extractvalue [[STRUCT2]][1]
    %0 = tt.reshape %arg {allow_reorder = true} : tensor<256xf32, #blocked0> -> tensor<256x1xf32,#blocked2>
    // CHECK:      [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
    // CHECK-NEXT: [[RES1:%.*]] = llvm.insertvalue [[T0]], [[RES]][0]
    // CHECK-NEXT: [[RES2:%.*]] = llvm.insertvalue [[T1]], [[RES1]][1]
    // CHECK-NEXT: [[RES3:%.*]] = llvm.insertvalue [[T0]], [[RES2]][2]
    // CHECK-NEXT: [[RES4:%.*]] = llvm.insertvalue [[T1]], [[RES3]][3]
    // CHECK-NEXT: [[RES5:%.*]] = llvm.insertvalue [[T0]], [[RES4]][4]
    // CHECK-NEXT: [[RES6:%.*]] = llvm.insertvalue [[T1]], [[RES5]][5]
    // CHECK-NEXT: [[RES7:%.*]] = llvm.insertvalue [[T0]], [[RES6]][6]
    // CHECK-NEXT: [[RES8:%.*]] = llvm.insertvalue [[T1]], [[RES7]][7]
    %1 = tt.broadcast %0 : tensor<256x1xf32,#blocked2> -> tensor<256x4xf32, #blocked2>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: basic_make_range
  tt.func @basic_make_range() {
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z12get_local_idj([[ZERO]]) {{.*}} : (i32) -> i64
    // CHECK:      [[RES:%.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
    // CHECK-NEXT: [[RES1:%.*]] = llvm.insertvalue {{.*}}, [[RES]][0]
    // CHECK-NEXT: [[RES2:%.*]] = llvm.insertvalue {{.*}}, [[RES1]][1]
    // CHECK-NEXT: [[RES3:%.*]] = llvm.insertvalue {{.*}}, [[RES2]][2]
    // CHECK-NEXT: [[RES4:%.*]] = llvm.insertvalue {{.*}}, [[RES3]][3]
    // CHECK-NEXT: [[RES5:%.*]] = llvm.insertvalue {{.*}}, [[RES4]][4]
    // CHECK-NEXT: [[RES6:%.*]] = llvm.insertvalue {{.*}}, [[RES5]][5]
    // CHECK-NEXT: [[RES7:%.*]] = llvm.insertvalue {{.*}}, [[RES6]][6]
    // CHECK-NEXT: [[RES8:%.*]] = llvm.insertvalue {{.*}}, [[RES7]][7]
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addf
  tt.func @basic_addf(%arg0 : tensor<256xf32,#blocked0>, %arg1 : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.fadd
    // CHECK: llvm.fadd
    %1 = arith.addf %arg0, %arg1 : tensor<256xf32,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addi
  tt.func @basic_addi(%arg0 : tensor<256xi32,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.add
    // CHECK: llvm.add
    %1 = arith.addi %arg0, %arg1 : tensor<256xi32,#blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_program_id
  tt.func @basic_program_id() {
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: [[GRP_ID:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[ZERO]]) {{.*}} : (i32) -> i64
    // CHECK-NEXT: llvm.trunc [[GRP_ID]] : i64 to i32
    %0 = tt.get_program_id x : i32
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_addptr
  tt.func @basic_addptr(%arg0 : tensor<256x!tt.ptr<f32>,#blocked0>, %arg1 : tensor<256xi32,#blocked0>) {
    // CHECK: llvm.getelementptr
    // CHECK: llvm.getelementptr
    %0 = tt.addptr %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_alloc_tensor(%arg0: !llvm.ptr<3>)
  tt.func @basic_alloc_tensor() {
    // CHECK-NEXT: llvm.mlir.constant
    // CHECK-NEXT: llvm.getelementptr
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_subview(%arg0: !llvm.ptr<3>)
  tt.func @basic_subview() {
    // CHECK: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc : () -> !tt.memdesc<128x16x32xf32, #shared0, #triton_gpu.shared_memory, mutable>
    %1 = triton_gpu.memdesc_subview %0[%index, %zero, %zero] : !tt.memdesc<128x16x32xf32, #shared0, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16x32xf32, #shared0, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----


#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: basic_splat
  tt.func @basic_splat(%ptr: !tt.ptr<f32>) {
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>,#blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_store
  tt.func @basic_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked0>, %vals: tensor<256xf32, #blocked0>, %mask: tensor<256xi1, #blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-DAG: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-DAG: [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f32, f32)>
    // CHECK-DAG: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(f32, f32)>
    // CHECK-DAG: [[ARG2_0:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(i1, i1)>
    // CHECK-DAG: [[ARG2_1:%.*]] = llvm.extractvalue %arg2[1] : !llvm.struct<(i1, i1)>

    // CHECK:      [[VEC1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast [[ARG1_0]] : f32 to f32
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[BCAST0]], [[VEC1]][[[CST_0]] : i32] : vector<1xf32>
    // CHECK-NEXT: [[BCAST1:%.*]] = llvm.bitcast [[IE1]] : vector<1xf32> to i32
    // CHECK-NEXT: [[AND1:%.*]] = llvm.and {{.*}}, [[ARG2_0]] : i1
    // CHECK-NEXT: [[VEC2:%.*]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: [[IE2:%.*]] = llvm.insertelement [[BCAST1]], [[VEC2]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK-NEXT: llvm.cond_br [[AND1]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store [[IE2]], [[BCAST2]] {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK:        [[AND2:%.*]] = llvm.and {{.*}}, [[ARG2_1]] : i1
    // CHECK-NEXT:   [[VEC3:%.*]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-NEXT:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:   [[IE3:%.*]] = llvm.insertelement {{.*}}, [[VEC3]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK:        llvm.cond_br [[AND2]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store [[IE3]], [[BCAST2]] {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb4
    // CHECK-NEXT: ^bb4:
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_layout_blocked_blocked({{.*}}%arg1: !llvm.ptr<3>)
  tt.func @convert_layout_blocked_blocked(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc  @_Z7barrierj([[ONE]]) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_layout_blocked_blocked_vec({{.*}}%arg1: !llvm.ptr<3>)
  tt.func @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z7barrierj([[ONE]]) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep({{.*}}%arg1: !llvm.ptr<3>)
  tt.func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: [[ONE_1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z7barrierj([[ONE_1]]) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 2], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas0 = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas0, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: convert_dot
  tt.func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.local_alloc %A : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory>
    %BB = triton_gpu.local_alloc %B : (tensor<16x16xf16, #blocked0>) -> !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory>
    %AA_DOT = triton_gpu.local_load %AA : !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory> -> tensor<16x16xf16, #dot_operand_a>
    %BB_DOT = triton_gpu.local_load %BB : !tt.memdesc<16x16xf16, #shared0, #triton_gpu.shared_memory> -> tensor<16x16xf16, #dot_operand_b>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #dpas0>

    // CHECK-COUNT-2: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #dpas0>

    tt.return
  }
}

// TODO: problems in MLIR's parser on slice layout
// #blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
// module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
//   tt.func @make_range_sliced_layout() {
//     %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked0}>>
//     tt.return
//   }
// }

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_layout_dpas_block
  tt.func @convert_layout_dpas_blocked(%arg0: tensor<32x16xf32, #dpas>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x16xf32, #dpas> -> tensor<32x16xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_layout_dpas_block
  tt.func @convert_layout_dpas_blocked(%arg0: tensor<32x64xf32, #dpas>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x64xf32, #dpas> -> tensor<32x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_layout_dpas_transpose
  tt.func @convert_layout_dpas_transpose(%arg0: tensor<128x256xf8E5M2, #dpas>) {
    // CHECK-COUNT-128: llvm.store %{{.*}} : vector<1xi8>, !llvm.ptr<3>
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK-COUNT-80: llvm.load %{{.*}} : !llvm.ptr<3> -> vector<1xi8>
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x256xf8E5M2, #dpas> -> tensor<128x256xf8E5M2, #blocked>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_layout_blocked_shared
  tt.func @convert_layout_blocked_shared(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<8xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<8xf32>, !llvm.ptr<3>
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice0
  tt.func @convert_blocked1d_to_slice0(%src:tensor<32xi32, #blocked0>) {
    // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
    %cvt = triton_gpu.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked1d_to_slice1
  tt.func @convert_blocked1d_to_slice1(%src:tensor<32xi32, #blocked0>) {
    // CHECK-COUNT-8: llvm.load {{.*}} : !llvm.ptr<3>
    %cvt = triton_gpu.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: convert_blocked_to_blocked_ptr
  tt.func @convert_blocked_to_blocked_ptr(%src:tensor<32x!tt.ptr<f32>, #blocked0>) {
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.store
    // CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
    // CHECK: llvm.inttoptr
    // CHECK-COUNT-4: llvm.insertvalue
    %cvt = triton_gpu.convert_layout %src : tensor<32x!tt.ptr<f32>, #blocked0> -> tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<128x32xf16, #shared>, %b:!tt.memdesc<32x256xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<128x32xf16, #shared> -> tensor<128x32xf16, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<32x256xf16, #shared> -> tensor<32x256xf16, #dot_operand_b>

    // CHECK-COUNT-128: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %28 = tt.dot %a_mat, %b_mat, %cst : tensor<128x32xf16, #dot_operand_a> * tensor<32x256xf16, #dot_operand_b> -> tensor<128x256xf32, #dpas>
    %38 = triton_gpu.convert_layout %28 : tensor<128x256xf32, #dpas> -> tensor<128x256xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul884_kernel_dot_operand_layout(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x64xf16, #shared0>, %b:!tt.memdesc<64x64xf16, #shared1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #dpas>
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x64xf16, #shared0> -> tensor<32x64xf16, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<64x64xf16, #shared1> -> tensor<64x64xf16, #dot_operand_b>

    // CHECK-COUNT-16: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    %28 = tt.dot %a_mat, %b_mat, %cst : tensor<32x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<32x64xf32, #dpas>
    %38 = triton_gpu.convert_layout %28 : tensor<32x64xf32, #dpas> -> tensor<32x64xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_fmadot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory>, %b:!tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    // CHECK: llvm.intr.fmuladd
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x16xf32, #shared, #triton_gpu.shared_memory> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<16x32xf32, #shared, #triton_gpu.shared_memory> -> tensor<16x32xf32, #dot_operand_b>

    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = ieee : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32dot
  tt.func @matmul_tf32dot(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a:!tt.memdesc<32x16xf32, #shared>, %b:!tt.memdesc<16x32xf32, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %a_mat = triton_gpu.local_load %a : !tt.memdesc<32x16xf32, #shared> -> tensor<32x16xf32, #dot_operand_a>
    %b_mat = triton_gpu.local_load %b : !tt.memdesc<16x32xf32, #shared> -> tensor<16x32xf32, #dot_operand_b>

    // CHECK-COUNT-2: llvm.call spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv8_fS_S_(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %28 = tt.dot %a_mat, %b_mat, %cst, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %38 = triton_gpu.convert_layout %28 : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>

    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.target" = "xpu", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z22__spirv_ControlBarrieriii(i32, i32, i32) attributes {convergent}
  // CHECK-LABEL: atomic_cas_f32_scalar_no_store
  tt.func @atomic_cas_f32_scalar_no_store(%ptr : !tt.ptr<f32>, %cmp : f32, %val : f32) {
    // CHECK:      [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[CMP0:%.*]] = llvm.icmp "eq"
    // CHECK:      [[MASK0:%.*]] = llvm.and [[TRUE]], [[CMP0]]
    // CHECK:      [[CMP:%.*]] = llvm.icmp "eq"
    // CHECK:      [[MASK:%.*]] = llvm.and [[MASK0]], [[CMP]]
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:      llvm.call spir_funccc @_Z22__spirv_ControlBarrieriii({{.*}}) {{.*}} : (i32, i32, i32) -> ()
    // CHECK-NEXT: llvm.cond_br [[MASK]], ^bb1, ^bb2([[ZERO]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast %arg1 : f32 to i32
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast %arg2 : f32 to i32
    // CHECK-NEXT:   [[CMPXCHG:%.*]] = llvm.cmpxchg %arg0, [[BCAST1]], [[BCAST2]] acq_rel monotonic : !llvm.ptr<1>, i32
    // CHECK-NEXT:   [[CMPXCHG_RES:%.*]] = llvm.extractvalue [[CMPXCHG]][0] : !llvm.struct<(i32, i1)>
    // CHECK-NEXT:   llvm.br ^bb2([[CMPXCHG_RES]] : i32)
    // CHECK-NEXT: ^bb2([[RES:%.*]]: i32):
    // CHECK-NEXT:   [[RES_CAST:%.*]] = llvm.bitcast [[RES]] : i32 to f32
    %0 = "tt.atomic_cas" (%ptr, %cmp, %val) {sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, f32) -> f32
    tt.return
  }
}

// -----

module attributes {"triton_gpu.target" = "xpu", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: atomic_cas_f32_scalar
  tt.func @atomic_cas_f32_scalar(%ptr : !tt.ptr<f32>, %cmp : f32, %val : f32) {
    // CHECK:      [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[CMP0:%.*]] = llvm.icmp "eq"
    // CHECK:      [[MASK0:%.*]] = llvm.and [[TRUE]], [[CMP0]]
    // CHECK:      [[CMP:%.*]] = llvm.icmp "eq"
    // CHECK:      [[MASK:%.*]] = llvm.and [[MASK0]], [[CMP]]
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.cond_br [[MASK]], ^bb1, ^bb2([[ZERO]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast %arg1 : f32 to i32
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast %arg2 : f32 to i32
    // CHECK-NEXT:   [[CMPXCHG:%.*]] = llvm.cmpxchg %arg0, [[BCAST1]], [[BCAST2]] acq_rel monotonic : !llvm.ptr<1>, i32
    // CHECK-NEXT:   [[CMPXCHG_RES:%.*]] = llvm.extractvalue [[CMPXCHG]][0] : !llvm.struct<(i32, i1)>
    // CHECK-NEXT:   llvm.br ^bb2([[CMPXCHG_RES]] : i32)
    // CHECK-NEXT: ^bb2([[RES:%.*]]: i32):
    // CHECK-NEXT:   [[RES_CAST:%.*]] = llvm.bitcast [[RES]] : i32 to f32
    // CHECK:        [[GEP:%.*]] = llvm.getelementptr %arg3[{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK-NEXT:   [[GEP_CAST:%.*]] = llvm.bitcast [[GEP]] : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT: llvm.cond_br [[MASK]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   llvm.store [[RES_CAST]], [[GEP_CAST]] : f32, !llvm.ptr<3>
    // CHECK-NEXT:   llvm.br ^bb4
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT    llvm.call spir_funccc @_Z7barrierj([[ONE]]) {{.*}} : (i32) -> ()
    %0 = "tt.atomic_cas" (%ptr, %cmp, %val) {sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, f32) -> f32
    tt.store %ptr, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z22__spirv_ControlBarrieriii(i32, i32, i32) attributes {convergent}
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK:      [[EV0_ARG2:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[EV1_ARG2:%.*]] = llvm.extractvalue %arg2[1] : !llvm.struct<(f32, f32)>
    // CHECK:      [[EV0_ARG0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[EV1_ARG0:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK:      [[CST_TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[UNDEF1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:      [[IE1:%.*]] = llvm.insertelement [[EV0_ARG2]], [[UNDEF1]][{{.*}} : i64] : vector<1xf32>
    // CHECK-NEXT: [[PRED1:%.*]] = llvm.and [[CST_TRUE]], {{.*}} : i1
    // CHECK-NEXT: [[ZERO1:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK:      llvm.cond_br [[PRED1]], ^bb1, ^bb2([[ZERO1]] : f32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE1]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES1:%.*]] = llvm.atomicrmw fadd [[EV0_ARG0]], [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb2([[RMW_RES1]] : f32)
    // CHECK-NEXT: ^bb2([[RMW_PHI1:%.*]]: f32):
    // CHECK-NEXT:   [[RMW_CAST:%.*]] = llvm.bitcast [[RMW_PHI1]] : f32 to f32
    // CHECK-NEXT:   [[UNDEF2:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:        [[IE2:%.*]] = llvm.insertelement [[EV1_ARG2]], [[UNDEF2]][{{.*}} : i64] : vector<1xf32>
    // CHECK-NEXT:   [[PRED2:%.*]] = llvm.and [[CST_TRUE]], {{.*}} : i1
    // CHECK-NEXT:   [[ZERO2:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK:        llvm.call spir_funccc @_Z22__spirv_ControlBarrieriii({{.*}}) {{.*}} : (i32, i32, i32) -> ()
    // CHECK-NEXT:   llvm.cond_br [[PRED2]], ^bb3, ^bb4([[ZERO2]] : f32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE2]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES2:%.*]] = llvm.atomicrmw fadd [[EV1_ARG0]], [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb4([[RMW_RES2]] : f32)
    // CHECK-NEXT: ^bb4([[RMW_PHI2:%.*]]: f32):
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.target" = "xpu", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z22__spirv_ControlBarrieriii(i32, i32, i32) attributes {convergent}
  // CHECK-LABEL: atomic_add_f32_scalar_no_store
  tt.func @atomic_add_f32_scalar_no_store(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK:      [[CST_TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[CMP:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND:%.*]] = llvm.and [[CST_TRUE]], [[CMP]]  : i1
    // CHECK:      [[CMP1:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND1:%.*]] = llvm.and [[AND]], [[CMP1]]  : i1
    // CHECK:      [[UNDEF1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:      [[IE1:%.*]] = llvm.insertelement %arg2, [[UNDEF1]][{{.*}} : i64] : vector<1xf32>
    // CHECK:      [[PRED:%.*]] = llvm.and [[AND1]], %arg1  : i1
    // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK:      llvm.call spir_funccc @_Z22__spirv_ControlBarrieriii({{.*}}) {{.*}} : (i32, i32, i32) -> ()
    // CHECK-NEXT: llvm.cond_br [[PRED]], ^bb1, ^bb2([[ZERO]] : f32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE1]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES:%.*]] = llvm.atomicrmw fadd %arg0, [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb2([[RMW_RES]] : f32)
    // CHECK-NEXT: ^bb2([[RMW_PHI:%.*]]: f32):
    // CHECK-NEXT:   [[RMW_CAST:%.*]] = llvm.bitcast [[RMW_PHI]] : f32 to f32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}

// -----

module attributes {"triton_gpu.target" = "xpu", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK:      [[CST_TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[CMP:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND:%.*]] = llvm.and [[CST_TRUE]], [[CMP]]  : i1
    // CHECK:      [[CMP1:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND1:%.*]] = llvm.and [[AND]], [[CMP1]]  : i1
    // CHECK:      [[UNDEF1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:      [[IE1:%.*]] = llvm.insertelement %arg2, [[UNDEF1]][{{.*}} : i64] : vector<1xf32>
    // CHECK:      [[PRED:%.*]] = llvm.and [[AND1]], %arg1  : i1
    // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK-NEXT: llvm.cond_br [[PRED]], ^bb1, ^bb2([[ZERO]] : f32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE1]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES:%.*]] = llvm.atomicrmw fadd %arg0, [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb2([[RMW_RES]] : f32)
    // CHECK-NEXT: ^bb2([[RMW_PHI:%.*]]: f32):
    // CHECK-NEXT:   [[RMW_CAST:%.*]] = llvm.bitcast [[RMW_PHI]] : f32 to f32
    // CHECK:        [[GEP:%.*]] = llvm.getelementptr %arg3[{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK-NEXT:   [[GEP_CAST:%.*]] = llvm.bitcast [[GEP]] : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT:   llvm.cond_br [[PRED]], ^bb3, ^bb4
    // CHECK-NEXT:  ^bb3:
    // CHECK-NEXT:    llvm.store [[RMW_CAST]], [[GEP_CAST]] : f32, !llvm.ptr<3>
    // CHECK-NEXT:    llvm.br ^bb4
    // CHECK-NEXT:  ^bb4:
    // CHECK-NEXT:    [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:    llvm.call spir_funccc @_Z7barrierj([[ONE]]) {{.*}} : (i32) -> ()
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32_sys_scope(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel : !llvm.ptr<1>, f32
    // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} acq_rel : !llvm.ptr<1>, f32
    %0 = tt.atomic_rmw fadd, relaxed, sys, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32
  tt.func @store_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xf32, #blocked0>) {
    // CHECK:      [[ARG0_0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(f32, f32)>
    // CHECK:      [[TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z12get_local_idj([[ZERO]]) {{.*}} : (i32) -> i64
    // CHECK:      llvm.cond_br [[TRUE]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store {{.*}}, [[BCAST]] {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK:        [[VEC:%.*]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-NEXT:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:   [[IE1:%.*]] = llvm.insertelement {{.*}}, [[VEC]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK:        llvm.cond_br [[TRUE]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store [[IE1]], [[BCAST1]] {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>

    tt.store %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_f32_scalar
  tt.func @store_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : f32) {
    // CHECK:      llvm.icmp "eq"
    // CHECK:      llvm.cond_br {{.*}}, ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST:%.*]] = llvm.bitcast %arg0 : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store {{.*}}, [[BCAST]] {alignment = 4 : i64} : vector<1xi32>, !llvm.ptr<1>
    tt.store %arg0, %arg1 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[GRP_ID_X:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[ZERO]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_X]] : i64 to i32
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[GRP_ID_Y:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[ONE]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_Y]] : i64 to i32
  // CHECK: [[TWO:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[GRP_ID_Z:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[TWO]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_Z]] : i64 to i32
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z: i32
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [4], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[GRP_ID_X:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[ZERO]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_X]] : i64 to i32
  // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[GRP_ID_Y:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[ONE]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_Y]] : i64 to i32
  // CHECK: [[TWO:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[GRP_ID_Z:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[TWO]]) {{.*}} : (i32) -> i64
  // CHECK: llvm.trunc [[GRP_ID_Z]] : i64 to i32
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z : i32
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_get_num_program
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[GRID_DIM_X:%.*]] = llvm.call spir_funccc @_Z14get_num_groupsj([[ZERO]]) {{.*}} : (i32) -> i64
    // CHECK: llvm.trunc [[GRID_DIM_X]] : i64 to i32
    // CHECK: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: [[GRID_DIM_Y:%.*]] = llvm.call spir_funccc @_Z14get_num_groupsj([[ONE]]) {{.*}} : (i32) -> i64
    // CHECK: llvm.trunc [[GRID_DIM_Y]] : i64 to i32
    // CHECK: [[TWO:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: [[GRID_DIM_Z:%.*]] = llvm.call spir_funccc @_Z14get_num_groupsj([[TWO]]) {{.*}} : (i32) -> i64
    // CHECK: llvm.trunc [[GRID_DIM_Z]] : i64 to i32
    %blockdimx = tt.get_num_programs x : i32
    %blockdimy = tt.get_num_programs y : i32
    %blockdimz = tt.get_num_programs z : i32
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32x!tt.ptr<i32>, #blocked0>

    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_index_cache
  tt.func @test_index_cache() {
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z12get_local_idj([[ZERO]]) {{.*}} : (i32) -> i64
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_base_index_cache
  tt.func @test_base_index_cache(%arg0: tensor<128x32xf32, #blocked0>) {
    // CHECK:      llvm.mlir.constant(0 : i32) : i32
    // CHECK:      llvm.mlir.constant(0 : i32) : i32
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z12get_local_idj([[ZERO]]) {{.*}} : (i32) -> i64
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    %1 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_index_cache_different_block
  tt.func @test_index_cache_different_block(%arg0: tensor<128x32xf32, #blocked0>, %arg1: i1) {
    // CHECK:      llvm.mlir.constant(0 : i32) : i32
    // CHECK:      llvm.mlir.constant(0 : i32) : i32
    // CHECK:      [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.call spir_funccc @_Z12get_local_idj([[ZERO]]) {{.*}} : (i32) -> i64
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
    cf.cond_br %arg1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = triton_gpu.local_alloc %arg0 : (tensor<128x32xf32, #blocked0>) -> !tt.memdesc<128x32xf32, #shared0, #triton_gpu.shared_memory>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      tt.return
  }
}

// -----

#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_tf32_cst_b
  tt.func @matmul_tf32_cst_b(%ptr:!tt.ptr<f32> {tt.divisibility = 16 : i32},
  %a: tensor<32x16xf32, #dot_operand_a>, %c: tensor<32x32xf32, #dpas>) {
  // CHECK: %[[CST:.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: %[[BC:.+]] = llvm.bitcast %[[CST]] : f32 to f32
  // CHECK: %[[SI:.+]] = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  // CHECK: llvm.insertvalue %[[BC]], %[[SI]][0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %b_mat = arith.constant dense<1.000000e+00> : tensor<16x32xf32, #dot_operand_b>
    %28 = tt.dot %a, %b_mat, %c, inputPrecision = tf32 : tensor<32x16xf32, #dot_operand_a> * tensor<16x32xf32, #dot_operand_b> -> tensor<32x32xf32, #dpas>
    %38 = triton_gpu.convert_layout %28 : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %36, %38 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_f16_cst_operands
  tt.func public @matmul_f16_cst_operands(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
  // CHECK: %[[C1f:.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16
  // CHECK: %[[Cf16:.+]] = llvm.bitcast %[[C1f]] : f16 to f16
  // CHECK: %[[U:.+]] = llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16, f16)>
  // CHECK: llvm.insertvalue %[[Cf16]], %[[U]][0]
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>>
    %cst_2 = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>> * tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>> -> tensor<32x32xf32, #dpas>
    %1 = triton_gpu.convert_layout %0 : tensor<32x32xf32, #dpas> -> tensor<32x32xf32, #blocked>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %4 = arith.muli %3, %cst_2 : tensor<32x1xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %9 = tt.broadcast %6 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %10 = tt.broadcast %8 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %11 = tt.addptr %9, %10 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    tt.store %11, %12 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: test_s8_to_bf16_conversion
  tt.func @test_s8_to_bf16_conversion(%in: tensor<32xi8, #blocked>) {
    // CHECK: llvm.sitofp
    // CHECK-NOT: llvm.sitofp
    %out = arith.sitofp %in : tensor<32xi8, #blocked> to tensor<32xbf16, #blocked>
    tt.return
  }
}

// -----
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
#dot = #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=4}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_intel_gpu.support_bf16_conversion"} {
  // CHECK-LABEL: test_s8_to_bf16_vectorized_conversion
  tt.func @test_s8_to_bf16_vectorized_conversion(%in: tensor<16x16xi8, #dpas>) {
     // CHECK: %[[F32:.+]] = llvm.sitofp %{{.*}} : i8 to f32
     // CHECK: %[[I16:.+]] = llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%{{.*}}) : (f32) -> i16
     // CHECK: llvm.bitcast %[[I16]] : i16 to bf16
     // CHECK-COUNT-15: llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%{{.*}}) : (f32) -> i16
    %out = arith.sitofp %in : tensor<16x16xi8, #dpas> to tensor<16x16xbf16, #dpas>
    tt.return
  }
}

// -----

// CHECK-LABEL: reduce_all
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice = #triton_gpu.slice<{dim = 0, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_all(%arg: tensor<256x1xi32, #blocked>, %arg_0: tensor<256x1xf32, #blocked>) {

    // CHECK: @_Z32sub_group_non_uniform_reduce_addf
    %0 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_addi
    %1 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.addi %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_mulf
    %2 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.mulf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_muli
    %3 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.muli %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_maxf
    %4 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.maxnumf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_minf
    %5 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.minnumf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_andi
    %6 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.andi %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z31sub_group_non_uniform_reduce_ori
    %7 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.ori %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z32sub_group_non_uniform_reduce_xori
    %8 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.xori %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    tt.return
  }
}

// -----

// CHECK-LABEL: reduce_cluster
#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice = #triton_gpu.slice<{dim = 0, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_cluster(%arg: tensor<256x1xi32, #blocked>, %arg_0: tensor<256x1xf32, #blocked>) {

    // CHECK: @_Z30sub_group_clustered_reduce_addfj
    %0 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_addij
    %1 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.addi %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_mulfj
    %2 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.mulf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_mulij
    %3 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.muli %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_maxfj
    %4 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.maxnumf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_minfj
    %5 = "tt.reduce"(%arg_0) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %48 = arith.minnumf %arg4, %arg5 : f32
      tt.reduce.return %48 : f32
    }) : (tensor<256x1xf32, #blocked>) -> tensor<1xf32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_andij
    %6 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.andi %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z29sub_group_clustered_reduce_orij
    %7 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.ori %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    // CHECK: @_Z30sub_group_clustered_reduce_xorij
    %8 = "tt.reduce"(%arg) <{axis = 0 : i32}> ({
    ^bb0(%arg4: i32, %arg5: i32):
      %48 = arith.xori %arg4, %arg5 : i32
      tt.reduce.return %48 : i32
    }) : (tensor<256x1xi32, #blocked>) -> tensor<1xi32, #slice>

    tt.return
  }
}

// -----

// CHECK-LABEL: sum_reduction
// CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_addi(%{{.*}}) {{.*}} : (i32) -> i32
// CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
// CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_addij(%{{.*}}, %{{.*}}) {{.*}}convergent{{.*}}no_unwind{{.*}}will_return{{.*}} : (i32, i32) -> i32

// CHECK: llvm.call spir_funccc @_Z7barrierj({{.*}}) {{.*}} : (i32) -> ()
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sum_reduction(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<1024> : tensor<1x1xi32, #blocked>
    %0 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #blocked1>
    %1 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xi32, #blocked>
    %3 = arith.muli %2, %cst : tensor<1x1xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<1x1x!tt.ptr<i32>, #blocked>, tensor<1x1xi32, #blocked>
    %6 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<1024xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x1024xi32, #blocked>
    %8 = tt.broadcast %5 : tensor<1x1x!tt.ptr<i32>, #blocked> -> tensor<1x1024x!tt.ptr<i32>, #blocked>
    %9 = tt.addptr %8, %7 : tensor<1x1024x!tt.ptr<i32>, #blocked>, tensor<1x1024xi32, #blocked>
    %10 = tt.load %9 : tensor<1x1024x!tt.ptr<i32>, #blocked>
    %11 = "tt.reduce"(%10) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %15 = arith.addi %arg2, %arg3 : i32
      tt.reduce.return %15 : i32
    }) : (tensor<1x1024xi32, #blocked>) -> tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = triton_gpu.convert_layout %11 : tensor<1xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<1xi32, #blocked1>
    %13 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked1>
    %14 = tt.addptr %13, %0 : tensor<1x!tt.ptr<i32>, #blocked1>, tensor<1xi32, #blocked1>
    tt.store %14, %12 : tensor<1x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice = #triton_gpu.slice<{dim = 1, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_bools({{.*}}, %arg1: !llvm.ptr<3>)
  tt.func public @reduce_bools(%arg: tensor<256x2xi1, #blocked>) {
    %24 = "tt.reduce"(%arg) <{axis = 1 : i32}> ({
    ^bb0(%arg4: i1, %arg5: i1):
      %48 = arith.ori %arg4, %arg5 : i1
      tt.reduce.return %48 : i1
    }) : (tensor<256x2xi1, #blocked>) -> tensor<256xi1, #slice>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_slice
//  CHECK-NOT: llvm.store {{.*}} !llvm.ptr<3>
//  CHECK-NOT: llvm.load {{.*}} !llvm.ptr<3>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 4, 2], warpsPerCTA = [2, 4, 2], order = [2, 0, 1], CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [0, 1, 2]}>
#sliced2 = #triton_gpu.slice<{dim = 2, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 16 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_slice() {
    %cst = arith.constant dense<true> : tensor<4x1xi1, #sliced2>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %1 = arith.ori %arg0, %arg1 : i1
      tt.reduce.return %1 : i1
    }) : (tensor<4x1xi1, #sliced2>) -> tensor<4xi1, #triton_gpu.slice<{dim = 1, parent = #sliced2}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: reduce_md_slice
//  CHECK: llvm.store {{.*}} !llvm.ptr<3>
//  CHECK: llvm.store {{.*}} !llvm.ptr<3>
//  CHECK: llvm.load {{.*}} !llvm.ptr<3>
//  CHECK: llvm.store {{.*}} !llvm.ptr<3>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#sliced = #triton_gpu.slice<{dim = 2, parent = #blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_md_slice(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x128xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %18 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %18 : f32
    }) {allocation.offset = 0 : i32} : (tensor<2x128xf32, #sliced>) -> tensor<2xf32, #triton_gpu.slice<{dim = 1, parent = #sliced}>>
    tt.return
  }
}

// -----

//  CHECK-LABEL: volta_dot
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @volta_dot() {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %a = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>>
    %b = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>>

    %87 = tt.dot %a, %b, %cst : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas, kWidth=2}>> * tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>> -> tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  tt.func public @convert_single_element() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %0 = triton_gpu.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: convert_single_element_and_add
  // CHECK-NOT: llvm.store
  // CHECK-NOT: llvm.load
  // CHECK: llvm.insertvalue
  // CHECK: llvm.extractvalue
  tt.func public @convert_single_element_and_add() attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked1>
    %cst2 = arith.constant dense<1.000000e+03> : tensor<1xf32, #blocked>
    %0 = triton_gpu.convert_layout %cst : tensor<1xf32, #blocked1> -> tensor<1xf32, #blocked>
    %1 = arith.addf %0, %cst2 : tensor<1xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_load
  // CHECK: llvm.load
  // CHECK-SAME: {alignment = 8 : i64} : !llvm.ptr<3> -> vector<8xi8>
  // CHECK-NOT: llvm.load
  tt.func public @vectorize_shmem_load(%shmem : !tt.memdesc<16x16xi8, #shared, #triton_gpu.shared_memory>) {
    %0 = triton_gpu.local_load %shmem : !tt.memdesc<16x16xi8, #shared, #triton_gpu.shared_memory> -> tensor<16x16xi8, #blocked>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @vectorize_shmem_store
  // CHECK: llvm.store
  // CHECK-SAME: {alignment = 64 : i64} : vector<16xi32>, !llvm.ptr<3>
  // CHECK-NOT: llvm.store
  tt.func public @vectorize_shmem_store(%block : tensor<64x64xi32, #blocked>) {
    %0 = triton_gpu.local_alloc %block : (tensor<64x64xi32, #blocked>) -> !tt.memdesc<64x64xi32, #shared, #triton_gpu.shared_memory>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: abs_is_int_min_poison
  // CHECK: %{{.*}} = "llvm.intr.abs"(%{{.*}}) <{is_int_min_poison = false}> : (i32) -> i32
  tt.func @abs_is_int_min_poison(%arg0 : tensor<256xi32, #blocked0>) {
    %abs = math.absi %arg0 : tensor<256xi32, #blocked0>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_load_bf16
  // CHECK: llvm.extractelement {{.*}} : vector<8xbf16>
  tt.func public @test_local_load_bf16() {
    %c0_i32 = arith.constant 0 : i32
    %19 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable>
    %22 = triton_gpu.memdesc_subview %19[%c0_i32, %c0_i32, %c0_i32] : !tt.memdesc<1x1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable>
    %39 = triton_gpu.local_load %22 : !tt.memdesc<1x2048xbf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<1x2048xbf16, #blocked>
    %40 = arith.extf %39 : tensor<1x2048xbf16, #blocked> to tensor<1x2048xf32, #blocked>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store
  // CHECK: llvm.store
  tt.func public @test_local_store(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc {allocation.offset = 0 : i32} : () -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %arg0, %0 : tensor<1xf32, #blocked> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: test_local_store_subview
  // CHECK: llvm.store
  tt.func public @test_local_store_subview(%arg0: tensor<1xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = triton_gpu.local_alloc {allocation.offset = 0 : i32} : () -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    %sv = triton_gpu.memdesc_subview %0[%c0_i32] : !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    triton_gpu.local_store %arg0, %sv : tensor<1xf32, #blocked> -> !tt.memdesc<1xf32, #shared, #triton_gpu.shared_memory, mutable>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: print_ptr
  tt.func @print_ptr(%arg0 : tensor<256x!tt.ptr<i32>, #blocked0>) {
    // CHECK: llvm.call @_Z18__spirv_ocl_printf(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} %{{.*}}) vararg(!llvm.func<i32 (ptr<2>, ...)>) : (!llvm.ptr<2>, i32, i32, i32, i32, !llvm.ptr<1>) -> i32
    // CHECK-NEXT: llvm.call @_Z18__spirv_ocl_printf(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) vararg(!llvm.func<i32 (ptr<2>, ...)>) : (!llvm.ptr<2>, i32, i32, i32, i32, !llvm.ptr<1>) -> i32
    tt.print "ptr: " {hex = false, isSigned = array<i32: 0>} : %arg0 : tensor<256x!tt.ptr<i32>, #blocked0>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // Test that %u format specifier is used if isSigned is false
  // CHECK: llvm.mlir.global internal constant @printfFormat_0("pid (%u, %u, %u) idx ()int32 tensor: %u\0A\00") {addr_space = 2 : i32}
  // CHECK-LABEL: print_int32_tensor_issigned_off
  // CHECK: llvm.call @_Z18__spirv_ocl_printf(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) vararg(!llvm.func<i32 (ptr<2>, ...)>) : (!llvm.ptr<2>, i32, i32, i32, i32) -> i32
  tt.func @print_int32_tensor_issigned_off(%arg0 : i32) {
    tt.print "int32 tensor: " {hex = false, isSigned = array<i32: 0>} : %arg0 : i32
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // Test that %i format specifier is used if isSigned is true
  // CHECK: llvm.mlir.global internal constant @printfFormat_0("pid (%u, %u, %u) idx ()int32 tensor: %i\0A\00") {addr_space = 2 : i32}
  // CHECK-LABEL: print_int32_tensor_issigned_on
  // CHECK: llvm.call @_Z18__spirv_ocl_printf(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) vararg(!llvm.func<i32 (ptr<2>, ...)>) : (!llvm.ptr<2>, i32, i32, i32, i32) -> i32
  tt.func @print_int32_tensor_issigned_on(%arg0 : i32) {
    tt.print "int32 tensor: " {hex = false, isSigned = array<i32: 1>} : %arg0 : i32
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_intel_gpu.support_bf16_conversion"} {
  tt.func @int32_to_bf16(%arg0: tensor<256xi32, #blocked>) {
    // CHECK-LABEL: @int32_to_bf16
    // CHECK: llvm.sitofp %{{.*}} : i32 to f32
    // CHECK-NEXT: llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%{{.*}}) : (f32) -> i16
    // CHECK-NEXT: llvm.bitcast %{{.*}} : i16 to bf16
    %a = arith.sitofp %arg0 : tensor<256xi32, #blocked> to tensor<256xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_intel_gpu.support_bf16_conversion} {
  tt.func @bf16_to_int32(%arg0: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: @bf16_to_int32
    // CHECK: llvm.bitcast %{{.*}} : bf16 to i16
    // CHECK-NEXT: llvm.call spir_funccc @_Z27__spirv_ConvertBF16ToFINTELs(%{{.*}}) : (i16) -> f32
    // CHECK-NEXT: llvm.fptosi %{{.*}} : f32 to i32
    %a = arith.fptosi %arg0 : tensor<256xbf16, #blocked> to tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @fp8_const(%arg0: tensor<1024xi1, #blocked>, %arg1: tensor<1024xf8E4M3FNUZ, #blocked>) {
    // CHECK-LABEL: @fp8_const
    // Copied from `tritongpu_to_llvm_hopper.mlir`
    // CHECK: llvm.mlir.constant(0.000000e+00 : f8E4M3FNUZ) : i8
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf8E4M3FNUZ, #blocked>
    %a = arith.select %arg0, %arg1, %cst : tensor<1024xi1, #blocked>, tensor<1024xf8E4M3FNUZ, #blocked>
    tt.return
  }
}
