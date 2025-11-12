// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: global_load_with_attributes
  tt.func @global_load_with_attributes(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : i32 -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xi32, #blocked0>
    %9 = tt.load %6 {isVolatile = true} : tensor<256x!tt.ptr<f32>, #blocked0>
    %10 = tt.load %6 cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    %12 = tt.load %6 cacheModifier = cg : tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.load %6 cacheModifier = wb : tensor<256x!tt.ptr<f32>, #blocked0>
    %14 = tt.load %6 cacheModifier = cs : tensor<256x!tt.ptr<f32>, #blocked0>
    %15 = tt.load %6 cacheModifier = wt : tensor<256x!tt.ptr<f32>, #blocked0>    
    %16 = tt.load %6 cacheModifier = cv : tensor<256x!tt.ptr<f32>, #blocked0>
    // CHECK-COUNT-2: llvm.load volatile {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64} : !llvm.ptr<1> -> vector<4xi32>    
    // CHECK-COUNT-2: llvm.load {{.*}} {alignment = 16 : i64, nontemporal} : !llvm.ptr<1> -> vector<4xi32>
    tt.return
  }
}
