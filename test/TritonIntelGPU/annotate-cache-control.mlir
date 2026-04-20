// RUN: triton-opt %s -split-input-file -tritonintelgpu-annotate-cache-control | FileCheck %s

// COM: Test a — load that does NOT feed a dot gets CG (cache = 2).

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_load_gets_cg
  tt.func public @streaming_load_gets_cg(%ptr: tensor<1024x!tt.ptr<f32>>, %out: tensor<1024x!tt.ptr<f32>>) {
    // CHECK: tt.load {{.*}} cacheModifier = cg
    %0 = tt.load %ptr : tensor<1024x!tt.ptr<f32>>
    // CHECK: tt.store {{.*}} cacheModifier = cg
    tt.store %out, %0 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test b — load that directly feeds a tt.dot stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_operand_load_unchanged
  tt.func public @gemm_operand_load_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                              %bptr: tensor<32x32x!tt.ptr<f16>>,
                                              %c: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
    tt.return %d : tensor<32x32xf32>
  }
}

// -----

// COM: Test c — load that feeds a tt.dot through an scf.for iter_arg stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_loop_iter_arg_unchanged
  tt.func public @gemm_loop_iter_arg_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                               %bptr: tensor<32x32x!tt.ptr<f16>>,
                                               %c: tensor<32x32xf32>,
                                               %lb: index, %ub: index, %step: index)
      -> tensor<32x32xf32> {
    // CHECK: tt.load
    // CHECK-NOT: cacheModifier = cg
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %c) -> tensor<32x32xf32> {
      %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
      %d = tt.dot %a, %b, %acc : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      scf.yield %d : tensor<32x32xf32>
    }
    tt.return %res : tensor<32x32xf32>
  }
}

// -----

// COM: Test d — store of a tt.dot result (GEMM epilogue) stays NONE.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @gemm_epilogue_store_unchanged
  tt.func public @gemm_epilogue_store_unchanged(%aptr: tensor<32x32x!tt.ptr<f16>>,
                                                %bptr: tensor<32x32x!tt.ptr<f16>>,
                                                %cptr: tensor<32x32x!tt.ptr<f32>>,
                                                %c: tensor<32x32xf32>) {
    %a = tt.load %aptr : tensor<32x32x!tt.ptr<f16>>
    %b = tt.load %bptr : tensor<32x32x!tt.ptr<f16>>
    %d = tt.dot %a, %b, %c : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
    // CHECK: tt.store
    // CHECK-NOT: cacheModifier = cg
    tt.store %cptr, %d : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test e — store of non-dot data gets CG.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @streaming_store_gets_cg
  tt.func public @streaming_store_gets_cg(%ptr: tensor<1024x!tt.ptr<f32>>,
                                          %val: tensor<1024xf32>) {
    // CHECK: tt.store {{.*}} cacheModifier = cg
    tt.store %ptr, %val : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
