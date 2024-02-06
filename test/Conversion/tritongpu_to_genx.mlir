// RUN: triton-opt %s -split-input-file --allocate-shared-memory="target=genx"  --convert-triton-gpu-to-llvm="target=genx" | FileCheck %s --implicit-check-not=llvm.inline_asm

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: llvm.func @test_empty_kernel(%arg0: i64, %arg1: !llvm.ptr<1>)
  // Here the 128 comes from the 4 in module attribute multiples 32
  // CHECK-SAME: attributes {genx.intel_reqd_sub_group_size = [32 : i32], genx.kernel = 1 : i32, genx.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
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
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i32
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i32] : vector<1xf32>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i32)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i32 to vector<1xf32>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf32>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i32)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i32 to vector<1xf32>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf32>
    // CHECK-NEXT:   [[RES1:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES2:%.*]] = llvm.insertvalue [[EE1]], [[RES1]][0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES3:%.*]] = llvm.insertvalue [[EE2]], [[RES2]][1] : !llvm.struct<(f32, f32)>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
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
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i32
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i32] : vector<1xf32>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i32)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i32 to vector<1xf32>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf32>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf32> to i32
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] : !llvm.ptr<1> -> i32
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i32)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i32):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i32 to vector<1xf32>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf32>
    // CHECK-NEXT:   [[RES1:%.*]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES2:%.*]] = llvm.insertvalue [[EE1]], [[RES1]][0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT:   [[RES3:%.*]] = llvm.insertvalue [[EE2]], [[RES2]][1] : !llvm.struct<(f32, f32)>

    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
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
    // CHECK-NEXT: [[CST_0:%.*]] = llvm.mlir.constant(0 : index) : i32
    // CHECK-NEXT: [[IE1:%.*]] = llvm.insertelement [[ARG2_0]], [[VEC]][[[CST_0]] : i32] : vector<1xf16>
    // CHECK-NEXT: [[BCAST0:%.*]] = llvm.bitcast [[IE1]] : vector<1xf16> to i16
    // CHECK:      llvm.cond_br [[ARG1_0]], ^bb1, ^bb2([[BCAST0]] : i16)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD1:%.*]] = llvm.load [[BCAST1]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb2([[LOAD1]] : i16)
    // CHECK-NEXT: ^bb2([[V1:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V1:%.*]] = llvm.bitcast [[V1]] : i16 to vector<1xf16>
    // CHECK:        [[EE1:%.*]] = llvm.extractelement [[BCAST_V1]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST2:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_1]], ^bb3, ^bb4([[BCAST2]] : i16)

    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST3:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD2:%.*]] = llvm.load [[BCAST3]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb4([[LOAD2]] : i16)
    // CHECK-NEXT: ^bb4([[V2:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V2:%.*]] = llvm.bitcast [[V2]] : i16 to vector<1xf16>
    // CHECK:        [[EE2:%.*]] = llvm.extractelement [[BCAST_V2]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST4:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_2]], ^bb5, ^bb6([[BCAST4]] : i16)

    // CHECK-NEXT: ^bb5:
    // CHECK-NEXT:   [[BCAST5:%.*]] = llvm.bitcast [[ARG0_2]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD3:%.*]] = llvm.load [[BCAST5]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb6([[LOAD3]] : i16)
    // CHECK-NEXT: ^bb6([[V3:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V3:%.*]] = llvm.bitcast [[V3]] : i16 to vector<1xf16>
    // CHECK:        [[EE3:%.*]] = llvm.extractelement [[BCAST_V3]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST5:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_3]], ^bb7, ^bb8([[BCAST5]] : i16)

    // CHECK-NEXT: ^bb7:
    // CHECK:        [[BCAST6:%.*]] = llvm.bitcast [[ARG0_3]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD4:%.*]] = llvm.load [[BCAST6]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb8([[LOAD4]] : i16)
    // CHECK-NEXT: ^bb8([[V4:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V4:%.*]] = llvm.bitcast [[V4]] : i16 to vector<1xf16>
    // CHECK:        [[EE4:%.*]] = llvm.extractelement [[BCAST_V4]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST7:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_4]], ^bb9, ^bb10([[BCAST7]] : i16)

    // CHECK-NEXT: ^bb9:
    // CHECK:        [[BCAST8:%.*]] = llvm.bitcast [[ARG0_4]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD5:%.*]] = llvm.load [[BCAST8]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb10([[LOAD5]] : i16)
    // CHECK-NEXT: ^bb10([[V5:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V5:%.*]] = llvm.bitcast [[V5]] : i16 to vector<1xf16>
    // CHECK:        [[EE5:%.*]] = llvm.extractelement [[BCAST_V5]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST8:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_5]], ^bb11, ^bb12([[BCAST8]] : i16)

    // CHECK-NEXT: ^bb11:
    // CHECK:        [[BCAST9:%.*]] = llvm.bitcast [[ARG0_5]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD6:%.*]] = llvm.load [[BCAST9]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb12([[LOAD6]] : i16)
    // CHECK-NEXT: ^bb12([[V6:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V6:%.*]] = llvm.bitcast [[V6]] : i16 to vector<1xf16>
    // CHECK:        [[EE6:%.*]] = llvm.extractelement [[BCAST_V6]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST10:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_6]], ^bb13, ^bb14([[BCAST10]] : i16)

    // CHECK-NEXT: ^bb13:
    // CHECK:        [[BCAST11:%.*]] = llvm.bitcast [[ARG0_6]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD7:%.*]] = llvm.load [[BCAST11]] : !llvm.ptr<1> -> i16
    // CHECK-NEXT:   llvm.br ^bb14([[LOAD7]] : i16)
    // CHECK-NEXT: ^bb14([[V7:%.*]]: i16):
    // CHECK-NEXT:   [[BCAST_V7:%.*]] = llvm.bitcast [[V7]] : i16 to vector<1xf16>
    // CHECK:        [[EE7:%.*]] = llvm.extractelement [[BCAST_V7]][{{.*}} : i32] : vector<1xf16>
    // CHECK:        [[BCAST12:%.*]] = llvm.bitcast {{.*}} : vector<1xf16> to i16
    // CHECK:        llvm.cond_br [[ARG1_7]], ^bb15, ^bb16([[BCAST12]] : i16)

    // CHECK-NEXT: ^bb15:
    // CHECK:        [[BCAST13:%.*]] = llvm.bitcast [[ARG0_7]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   [[LOAD8:%.*]] = llvm.load [[BCAST13]] : !llvm.ptr<1> -> i16
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
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf16, #blocked0>
    tt.return
  }
}

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
    // CHECK:      genx.workitem.id.x : i32
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
    // CHECK: genx.workgroup.id.x : i32
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
    // CHECK-NEXT: llvm.bitcast
    %0 = triton_gpu.alloc_tensor : tensor<16x16xf16, #shared0>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_extract_slice(%arg0: !llvm.ptr<3>)
  tt.func @basic_extract_slice() {
    // CHECK: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.extractvalue
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.mul
    // CHECK-NEXT: llvm.add
    // CHECK-NEXT: llvm.getelementptr
    %index = arith.constant 1 : i32
    %0 = triton_gpu.alloc_tensor : tensor<128x16x32xf32, #shared0>
    %1 = triton_gpu.extract_slice %0[%index, 0, 0][1, 16, 32][1, 1, 1] : tensor<128x16x32xf32, #shared0> to tensor<16x32xf32, #shared0>
    tt.return
  }
}

// -----

#block0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#block2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#block3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#slice2d1 = #triton_gpu.slice<{dim = 1, parent=#block2}>
#slice3d0 = #triton_gpu.slice<{dim = 0, parent=#block3}>
#AL = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_insert_slice_async_fallback
  tt.func @basic_insert_slice_async_fallback(%arg0: !tt.ptr<f16> {tt.divisibility = 1 : i32}) {
    %off0_ = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #slice2d1>
    %off1_ = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice3d0>
    %off0 = tt.expand_dims %off0_ {axis = 1 : i32} : tensor<16xi32, #slice2d1> -> tensor<16x1xi32, #block2>
    %off1 = tt.expand_dims %off1_ {axis = 0 : i32} : tensor<64xi32, #slice3d0> -> tensor<1x64xi32, #block3>
    %broadcast_off0_scalar = tt.broadcast %off0 : tensor<16x1xi32, #block2> -> tensor<16x64xi32, #block2>
    %cst_scalar = arith.constant 64 : i32
    %cst = tt.splat %cst_scalar : i32 -> tensor<16x64xi32, #block2>
    %broadcast_off0_ = arith.muli %broadcast_off0_scalar, %cst : tensor<16x64xi32, #block2>
    %broadcast_off1_ = tt.broadcast %off1 : tensor<1x64xi32, #block3> -> tensor<16x64xi32, #block3>
    %broadcast_off0 = triton_gpu.convert_layout %broadcast_off0_ : tensor<16x64xi32, #block2> -> tensor<16x64xi32, #AL>
    %broadcast_off1 = triton_gpu.convert_layout %broadcast_off1_ : tensor<16x64xi32, #block3> -> tensor<16x64xi32, #AL>
    %off = arith.addi %broadcast_off0, %broadcast_off1 : tensor<16x64xi32, #AL>
    %a_init = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>, #AL>
    %a_ptr = tt.addptr %a_init, %off : tensor<16x64x!tt.ptr<f16>, #AL>, tensor<16x64xi32, #AL>
    %tensor = triton_gpu.alloc_tensor : tensor<2x16x64xf16, #A>
    %index = arith.constant 1 : i32

    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<8xi32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<8xi32>
    %a = triton_gpu.insert_slice_async %a_ptr, %tensor, %index {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64x!tt.ptr<f16>, #AL> -> tensor<2x16x64xf16, #A>
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
    // CHECK-NEXT: [[ARG0_1:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[ARG1_0:%.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[ARG1_1:%.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[ARG2_0:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(i1, i1)>
    // CHECK-NEXT: [[ARG2_1:%.*]] = llvm.extractvalue %arg2[1] : !llvm.struct<(i1, i1)>

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
    // CHECK-NEXT:   llvm.store [[IE2]], [[BCAST2]] : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK:        [[AND2:%.*]] = llvm.and {{.*}}, [[ARG2_1]] : i1
    // CHECK-NEXT:   [[VEC3:%.*]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-NEXT:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:   [[IE3:%.*]] = llvm.insertelement {{.*}}, [[VEC3]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK:        llvm.cond_br [[AND2]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store [[IE3]], [[BCAST2]] : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb4
    // CHECK-NEXT: ^bb4:
    tt.store %ptrs, %vals, %mask : tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
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
    // CHECK: genx.barrier
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
  // CHECK-LABEL: convert_layout_blocked_blocked_vec({{.*}}%arg1: !llvm.ptr<3>)
  tt.func @convert_layout_blocked_blocked_vec(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: genx.barrier
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
  // CHECK-LABEL: convert_layout_blocked_blocked_multi_rep({{.*}}%arg1: !llvm.ptr<3>)
  tt.func @convert_layout_blocked_blocked_multi_rep(%arg0: tensor<16x16xf32, #blocked0>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: genx.barrier
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: genx.barrier
    // CHECK: llvm.store
    // CHECK-SAME: vector<4xf32>, !llvm.ptr<3>
    // CHECK: genx.barrier
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x16xf32, #blocked0> -> tensor<16x16xf32, #blocked1>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_f16_f16_f32_1
  tt.func @dot_f32_f16_f16_f32_1(%a: tensor<8x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<8x16xf32, #mma>) {
    // CHECK: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<FP16>, pb = #genx.precision_type<FP16>, rc = 8 : i32} : (vector<8xf32>, vector<4xi32>, vector<8xi32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<8x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<8x16xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_f16_f16_f32_2
  tt.func @dot_f32_f16_f16_f32_2(%a: tensor<16x16xf16, #dot_operand_a>, %b: tensor<16x16xf16, #dot_operand_b>, %c: tensor<16x16xf32, #mma>) {
    // COM: 2 repetitions along axis for M.
    // CHECK-COUNT-2: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<FP16>, pb = #genx.precision_type<FP16>, rc = 8 : i32} : (vector<8xf32>, vector<4xi32>, vector<8xi32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_i32_i8_i8_i32_1
  tt.func @dot_i32_i8_i8_i32_1(%a: tensor<8x32xi8, #dot_operand_a>, %b: tensor<32x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #mma>) {
    // CHECK: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<S8>, pb = #genx.precision_type<S8>, rc = 8 : i32} : (vector<8xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<8x32xi8, #dot_operand_a> * tensor<32x16xi8, #dot_operand_b> -> tensor<8x16xi32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_i32_i8_i8_i32_2
  tt.func @dot_i32_i8_i8_i32_2(%a: tensor<8x64xi8, #dot_operand_a>, %b: tensor<64x16xi8, #dot_operand_b>, %c: tensor<8x16xi32, #mma>) {
    // COM: 2 repetition along axis for K.
    // CHECK: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<S8>, pb = #genx.precision_type<S8>, rc = 8 : i32} : (vector<8xi32>, vector<4xi32>, vector<8xi32>) -> vector<8xi32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<8x64xi8, #dot_operand_a> * tensor<64x16xi8, #dot_operand_b> -> tensor<8x16xi32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_tf32_tf32_f32_1
  tt.func @dot_f32_tf32_tf32_f32_1(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x16xf32, #dot_operand_b>, %c: tensor<8x16xf32, #mma>) {
    // CHECK: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<TF32>, pb = #genx.precision_type<TF32>, rc = 8 : i32} : (vector<8xf32>, vector<4xi32>, vector<8xi32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<8x8xf32, #dot_operand_a> * tensor<8x16xf32, #dot_operand_b> -> tensor<8x16xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: dot_f32_tf32_tf32_f32_2
  tt.func @dot_f32_tf32_tf32_f32_2(%a: tensor<8x8xf32, #dot_operand_a>, %b: tensor<8x32xf32, #dot_operand_b>, %c: tensor<8x32xf32, #mma>) {
    // COM: 2 repetitions along axis for N.
    // CHECK-COUNT-2: genx.matrix.dpas {{.*}}, {{.*}}, {{.*}} {pa = #genx.precision_type<TF32>, pb = #genx.precision_type<TF32>, rc = 8 : i32} : (vector<8xf32>, vector<4xi32>, vector<8xi32>) -> vector<8xf32>
    %0 = tt.dot %a, %b, %c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<8x8xf32, #dot_operand_a> * tensor<8x32xf32, #dot_operand_b> -> tensor<8x32xf32, #mma>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_dpas_block
  tt.func @convert_layout_dpas_blocked(%arg0: tensor<32x16xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: genx.barrier
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x16xf32, #mma> -> tensor<32x16xf32, #blocked0>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: convert_layout_dpas_block
  tt.func @convert_layout_dpas_blocked(%arg0: tensor<32x64xf32, #mma>) {
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: llvm.store
    // CHECK-SAME: vector<1xf32>, !llvm.ptr<3>
    // CHECK: genx.barrier
    // CHECK: llvm.load
    // CHECK-SAME: !llvm.ptr<3> -> vector<4xf32>
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x64xf32, #mma> -> tensor<32x64xf32, #blocked>
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
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x32xf32, #blocked0> -> tensor<128x32xf32, #shared0>
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
    // CHECK-COUNT-8: llvm.load {{.*}} : !llvm.ptr<3> -> vector<1xi32>
    %cvt = triton_gpu.convert_layout %src : tensor<32xi32, #blocked0> -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_blocked_to_blocked_ptr
  tt.func @convert_blocked_to_blocked_ptr(%src:tensor<32x!tt.ptr<f32>, #blocked0>) {
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.store
    // CHECK: genx.barrier
    // CHECK: llvm.inttoptr
    // CHECK-COUNT-4: llvm.insertvalue
    %cvt = triton_gpu.convert_layout %src : tensor<32x!tt.ptr<f32>, #blocked0> -> tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_cas_f32_scalar
  // CHECK-SAME:    ({{.*}}, [[SMEM:%.*]]: !llvm.ptr<3>)
  tt.func @atomic_cas_f32_scalar(%ptr: !tt.ptr<f32>, %cmp: f32, %val: f32) {
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
    // CHECK-NEXT:   genx.barrier
    // CHECK-NEXT:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:   [[GEP:%.*]] = llvm.getelementptr %arg3[[[ZERO]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    // CHECK-NEXT:   [[BCAST:%.*]] = llvm.bitcast [[GEP]] : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT:   llvm.cond_br [[MASK]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:  // pred: ^bb2
    // CHECK-NEXT:   llvm.store [[RES_CAST]], [[BCAST]] : f32, !llvm.ptr<3>
    // CHECK-NEXT:   llvm.br ^bb4
    // CHECK-NEXT: ^bb4:
    // CHECK-NEXT:   genx.barrier
    // CHECK-NEXT:   {{.*}} = llvm.load [[BCAST]] : !llvm.ptr<3> -> f32
    // CHECK-NEXT:   genx.barrier
    %0 = "tt.atomic_cas" (%ptr, %cmp, %val) {sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, f32) -> f32
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK:      [[EV0_ARG2:%.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(f32, f32)>
    // CHECK-NEXT: [[EV1_ARG2:%.*]] = llvm.extractvalue %arg2[1] : !llvm.struct<(f32, f32)>
    // CHECK:      [[EV0_ARG0:%.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK-NEXT: [[EV1_ARG0:%.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK:      [[CST_TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[UNDEF1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:      [[IE1:%.*]] = llvm.insertelement [[EV0_ARG2]], [[UNDEF1]][{{.*}} : i32] : vector<1xf32>
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
    // CHECK:        [[IE2:%.*]] = llvm.insertelement [[EV1_ARG2]], [[UNDEF2]][{{.*}} : i32] : vector<1xf32>
    // CHECK-NEXT:   [[PRED2:%.*]] = llvm.and [[CST_TRUE]], {{.*}} : i1
    // CHECK-NEXT:   [[ZERO2:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK-NEXT:   llvm.cond_br [[PRED2]], ^bb3, ^bb4([[ZERO2]] : f32)
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE2]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES2:%.*]] = llvm.atomicrmw fadd [[EV1_ARG0]], [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb4([[RMW_RES2]] : f32)
    // CHECK-NEXT: ^bb4([[RMW_PHI2:%.*]]: f32):
    %0 = "tt.atomic_rmw" (%arg0, %arg2, %arg1) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.return
  }
}

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  // CHECK-SAME:    ({{.*}}, [[SMEM:%.*]]: !llvm.ptr<3>)
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK:      [[CST_TRUE:%.*]] = llvm.mlir.constant(true) : i1
    // CHECK:      [[CMP:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND:%.*]] = llvm.and [[CST_TRUE]], [[CMP]]  : i1
    // CHECK:      [[CMP1:%.*]] = llvm.icmp "eq"
    // CHECK-NEXT: [[AND1:%.*]] = llvm.and [[AND]], [[CMP1]]  : i1
    // CHECK:      [[UNDEF1:%.*]] = llvm.mlir.undef : vector<1xf32>
    // CHECK:      [[IE1:%.*]] = llvm.insertelement %arg2, [[UNDEF1]][{{.*}} : i32] : vector<1xf32>
    // CHECK:      [[PRED:%.*]] = llvm.and [[AND1]], %arg1  : i1
    // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
    // CHECK:      llvm.cond_br [[PRED]], ^bb1, ^bb2([[ZERO]] : f32)
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST2:%.*]] = llvm.bitcast [[IE1]] : vector<1xf32> to f32
    // CHECK-NEXT:   [[RMW_RES:%.*]] = llvm.atomicrmw fadd %arg0, [[BCAST2]] acq_rel : !llvm.ptr<1>, f32
    // CHECK-NEXT:   llvm.br ^bb2([[RMW_RES]] : f32)
    // CHECK-NEXT: ^bb2([[RMW_PHI:%.*]]: f32):
    // CHECK-NEXT:   [[RMW_CAST:%.*]] = llvm.bitcast [[RMW_PHI]] : f32 to f32
    // CHECK:        [[GEP:%.*]] = llvm.getelementptr [[SMEM]][{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>
    // CHECK-NEXT:   [[BCAST_GEP:%.*]] = llvm.bitcast [[GEP]] : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK-NEXT:   llvm.cond_br [[PRED]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:    llvm.store [[RMW_CAST]], [[BCAST_GEP]] : f32, !llvm.ptr<3>
    // CHECK-NEXT:    llvm.br ^bb4
    // CHECK-NEXT:  ^bb4:
    // CHECK-NEXT:    genx.barrier
    %0 = "tt.atomic_rmw" (%arg0, %arg2, %arg1) {atomic_rmw_op = 5 : i32, sem = 1 : i32, scope = 1 : i32} : (!tt.ptr<f32>, f32, i1) -> f32
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
    // CHECK:      genx.workitem.id.x : i32
    // CHECK:      llvm.cond_br [[TRUE]], ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT:   [[BCAST:%.*]] = llvm.bitcast [[ARG0_0]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store {{.*}}, [[BCAST]] : vector<1xi32>, !llvm.ptr<1>
    // CHECK-NEXT:   llvm.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK:        [[VEC:%.*]] = llvm.mlir.undef : vector<1xi32>
    // CHECK-NEXT:   [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:   [[IE1:%.*]] = llvm.insertelement {{.*}}, [[VEC]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK:        llvm.cond_br [[TRUE]], ^bb3, ^bb4
    // CHECK-NEXT: ^bb3:
    // CHECK-NEXT:   [[BCAST1:%.*]] = llvm.bitcast [[ARG0_1]] : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK-NEXT:   llvm.store [[IE1]], [[BCAST1]] : vector<1xi32>, !llvm.ptr<1>

    tt.store %arg0, %arg1 : tensor<256xf32, #blocked0>
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
    // CHECK-NEXT:   llvm.store {{.*}}, [[BCAST]] : vector<1xi32>, !llvm.ptr<1>
    tt.store %arg0, %arg1 : f32
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
// CHECK-LABEL: test_get_program_id
tt.func @test_get_program_id(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
  // CHECK: genx.workgroup.id.x
  // CHECK: genx.workgroup.id.y
  // CHECK: genx.workgroup.id.z
  %blockidx = tt.get_program_id x: i32
  %blockidy = tt.get_program_id y: i32
  %blockidz = tt.get_program_id z : i32
  %v0 = arith.addi %blockidx, %blockidy : i32
  %v1 = arith.addi %v0, %blockidz : i32
  %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
  tt.store %a, %0 : tensor<32xi32, #blocked0>

  tt.return
}

}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_get_num_program
  tt.func @test_get_num_program(%a: tensor<32x!tt.ptr<i32>, #blocked0>) {
    // CHECK: genx.grid.dim.x
    // CHECK: genx.grid.dim.y
    // CHECK: genx.grid.dim.z
    %blockdimx = tt.get_num_programs {axis=0:i32} : i32
    %blockdimy = tt.get_num_programs {axis=1:i32} : i32
    %blockdimz = tt.get_num_programs {axis=2:i32} : i32
    %v0 = arith.addi %blockdimx, %blockdimy : i32
    %v1 = arith.addi %v0, %blockdimz : i32
    %0 = tt.splat %v1 : i32 -> tensor<32xi32, #blocked0>
    tt.store %a, %0 : tensor<32xi32, #blocked0>

    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: test_index_cache
  tt.func @test_index_cache() {
    // CHECK: genx.workitem.id.x
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
    // CHECK: genx.workitem.id.x
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x32xf32, #blocked0> -> tensor<128x32xf32, #shared0>
    %1 = triton_gpu.convert_layout %arg0 : tensor<128x32xf32, #blocked0> -> tensor<128x32xf32, #shared0>
    tt.return
  }
}

// -----
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: test_index_cache_different_block
  tt.func @test_index_cache_different_block(%arg0: tensor<128x32xf32, #blocked0>, %arg1: i1) {
    // CHECK: genx.workitem.id.x
    %0 = triton_gpu.convert_layout %arg0 : tensor<128x32xf32, #blocked0> -> tensor<128x32xf32, #shared0>
    cf.cond_br %arg1, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %1 = triton_gpu.convert_layout %arg0 : tensor<128x32xf32, #blocked0> -> tensor<128x32xf32, #shared0>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_reduce_1
  tt.func @basic_reduce_1(%f : tensor<4xi32, #blocked>) {
    // FIXME: Merge 1c45836 regression
    %g = "tt.reduce" (%f) ({
    ^bb0(%arg0: i32, %arg1: i32):
      %add = arith.addi %arg0, %arg1 : i32
      tt.reduce.return %add : i32
    }) {axis = 0 : i32} : (tensor<4xi32, #blocked>) -> i32
    tt.return
  }
}
