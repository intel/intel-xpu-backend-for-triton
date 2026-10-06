// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: Issue #7540: the minimum element count in the convert cost model models
// COM: the SLM round-trip granularity, so it must not be applied to a convert
// COM: that merely reorders registers. Applying it priced this single-element
// COM: reduction-result convert as a full SLM round-trip, which made backward
// COM: rematerialization duplicate the global load and the reduction in order to
// COM: remove an almost-free convert. The load and the reduce must therefore
// COM: stay single and the convert must be preserved.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: @remat_free_convert_not_beneficial
  // CHECK: tt.load
  // CHECK-NOT: tt.load
  // CHECK: "tt.reduce"
  // CHECK-NOT: "tt.reduce"
  // CHECK: ttg.convert_layout
  // CHECK: tt.return
  tt.func public @remat_free_convert_not_beneficial(%inp: !tt.ptr<bf16>, %oa: !tt.ptr<f32>, %ob: !tt.ptr<f32>) {
    %rng = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %ed = tt.expand_dims %rng {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %sp = tt.splat %inp : !tt.ptr<bf16> -> tensor<1x512x!tt.ptr<bf16>, #blocked>
    %pp = tt.addptr %sp, %ed : tensor<1x512x!tt.ptr<bf16>, #blocked>, tensor<1x512xi32, #blocked>
    %a = tt.load %pp {ttig.block_io = "row_major"} : tensor<1x512x!tt.ptr<bf16>, #blocked>
    %f = arith.extf %a : tensor<1x512xbf16, #blocked> to tensor<1x512xf32, #blocked>
    // COM: A max-style reduce is required: a plain associative addf reduce is
    // COM: rejected earlier by the non-associative check and never reaches the
    // COM: cost model.
    %r = "tt.reduce"(%f) <{axis = 1 : i32}> ({
    ^bb0(%x: f32, %y: f32):
      %m = arith.cmpf ogt, %x, %y : f32
      %sel = arith.select %m, %x, %y : f32
      tt.reduce.return %sel : f32
    }) : (tensor<1x512xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // COM: Consuming the reduction result in two layouts makes the slice
    // COM: rematerializable, which is what reaches the cost model.
    %c = ttg.convert_layout %r : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %eb = tt.expand_dims %c {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x1xf32, #blocked1>
    %sb = tt.splat %ob : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked1>
    tt.store %sb, %eb : tensor<1x1x!tt.ptr<f32>, #blocked1>
    %ea = tt.expand_dims %r {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1xf32, #blocked>
    %sa = tt.splat %oa : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked>
    tt.store %sa, %ea : tensor<1x1x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
