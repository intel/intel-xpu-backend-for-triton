// RUN: LD_PRELOAD=%shlibdir/../plugins/libtriton.so \
// RUN: TRITON_PLUGIN_PATHS=%shlibdir/../plugins/libMLIRDialectPlugin.so \
// RUN: triton-opt \
// RUN: -split-input-file --convert-plugin-gpu-to-llvm --convert-triton-gpu-to-llvm %s | \
// RUN: FileCheck %s

<<<<<<< HEAD
// REQUIRES: plugins, shared-libs
=======
// REQUIRES: triton-ext-enabled
// XFAIL: *
>>>>>>> 6cb844590c72cd8e0e827007c2d5cd04fc64b1bb

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func @convert_plugin() {
    // CHECK-DAG: %[[THREADIDX:.*]] = nvvm.read.ptx.sreg.tid.x
    %0 = arith.constant 0 : i32
    %1 = plugin.magic %0 : i32
    %2 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    tt.return
  }
}
