// RUN: triton-opt %s -split-input-file -tritonintelgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=true" -canonicalize | FileCheck %s

// COM: Test 1 - Simple case: pid-based offset + lane range, single load.
// COM: The pass hoists the uniform splat(pid_offset) component into the
// COM: scalar base via a scalar tt.addptr, and leaves only the non-uniform
// COM: lane range in the tensor offset.  The loop carries two scalar
// COM: !tt.ptr<f32> iter-args (one per input pointer) instead of two
// COM: tensor<1024x!tt.ptr<f32>>.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @simple_pid_offset(
  tt.func public @simple_pid_offset(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %stride: i32, %n_steps: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %pid_offset = arith.muli %pid, %c1024_i32 : i32
    %lane_range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %pid_splat = tt.splat %pid_offset : i32 -> tensor<1024xi32>
    %total_offset = arith.addi %pid_splat, %lane_range : tensor<1024xi32>
    %base_a = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr_a = tt.addptr %base_a, %total_offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %base_b = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr_b = tt.addptr %base_b, %total_offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %base_c = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr_c = tt.addptr %base_c, %total_offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %stride_tensor = tt.splat %stride : i32 -> tensor<1024xi32>
    // COM: After transformation: iter_args carry scalar !tt.ptr<f32> for each
    // COM: input pointer.
    // CHECK: scf.for {{.*}} iter_args(%[[ITER_A:.*]] = %{{.*}}, %[[ITER_B:.*]] = %{{.*}}) -> (!tt.ptr<f32>, !tt.ptr<f32>)
    %result:2 = scf.for %iv = %c0_i32 to %n_steps step %c1_i32
        iter_args(%arg_a = %ptr_a, %arg_b = %ptr_b)
        -> (tensor<1024x!tt.ptr<f32>>, tensor<1024x!tt.ptr<f32>>) : i32 {
      // COM: Inside loop: materialize via splat(scalar_base) + lane_range.
      // CHECK: tt.splat %[[ITER_A]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      // CHECK: tt.addptr {{.*}} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %ld_a = tt.load %arg_a : tensor<1024x!tt.ptr<f32>>
      %ld_b = tt.load %arg_b : tensor<1024x!tt.ptr<f32>>
      %sum = arith.addf %ld_a, %ld_b : tensor<1024xf32>
      tt.store %ptr_c, %sum : tensor<1024x!tt.ptr<f32>>
      // COM: Loop advance uses scalar tt.addptr(scalar_base, scalar_stride).
      // CHECK: tt.addptr %[[ITER_A]], %{{.*}} : !tt.ptr<f32>, i32
      // CHECK: tt.addptr %[[ITER_B]], %{{.*}} : !tt.ptr<f32>, i32
      %next_a = tt.addptr %arg_a, %stride_tensor : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %next_b = tt.addptr %arg_b, %stride_tensor : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      scf.yield %next_a, %next_b : tensor<1024x!tt.ptr<f32>>, tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// COM: Test 2 - Loop with dense<1024> constant stride.
// COM: The pass recognizes the splatted constant stride and promotes the loop
// COM: pointer to a scalar base.  The iter_arg carries only !tt.ptr<f32>;
// COM: the non-uniform lane range is re-materialized each iteration.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @loop_dense_stride(
  tt.func @loop_dense_stride(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c1024 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %pid_offset = arith.muli %pid, %c1024 : i32
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %pid_t = tt.splat %pid_offset : i32 -> tensor<1024xi32>
    %total = arith.addi %pid_t, %range : tensor<1024xi32>
    %base_a = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr_a = tt.addptr %base_a, %total : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %base_out = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr_out = tt.addptr %base_out, %total : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // COM: dense<1024> is a splatted constant: the pass hoists it as a scalar
    // COM: stride on the base pointer each iteration.
    %stride = arith.constant dense<1024> : tensor<1024xi32>
    // COM: After transformation: iter_arg carries just !tt.ptr<f32>.
    // CHECK: scf.for {{.*}} iter_args(%[[BASE:.*]] = %{{.*}}) -> (!tt.ptr<f32>)
    scf.for %iv = %c0 to %n step %c1 iter_args(%arg_a = %ptr_a) -> (tensor<1024x!tt.ptr<f32>>) : i32 {
      // CHECK: tt.splat %[[BASE]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      // CHECK: tt.addptr {{.*}} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // CHECK: tt.load
      %val = tt.load %arg_a : tensor<1024x!tt.ptr<f32>>
      tt.store %ptr_out, %val : tensor<1024x!tt.ptr<f32>>
      // COM: Scalar base advances by 1024 each iteration.
      // CHECK: tt.addptr %[[BASE]], %{{.*}} : !tt.ptr<f32>, i32
      %next_a = tt.addptr %arg_a, %stride : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      scf.yield %next_a : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

// COM: Test 3 - Descriptor-load kernel: the pass initializes the scalar !tt.ptr
// COM: arg with an unrealized_cast, but neither the unrealized_cast nor the
// COM: descriptor_load result has any uses, so canonicalize removes both.
// COM: The final function body is just tt.return.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @descriptor_load_no_op(
  tt.func public @descriptor_load_no_op(%desc: !tt.tensordesc<1024xf32>, %out: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    // COM: Both the unrealized_cast (for %out) and the dead descriptor_load
    // COM: are eliminated by canonicalize, leaving only tt.return.
    // CHECK-NOT: builtin.unrealized_conversion_cast
    // CHECK-NOT: tt.descriptor_load
    // CHECK: tt.return
    %val = tt.descriptor_load %desc[%c0_i32] : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    tt.return
  }
}

// -----

// COM: Test 4 - Regression test for #7435: an i64-typed offset intermediate
// COM: (e.g. as emitted by PyTorch Inductor to avoid i32 overflow, hence
// COM: "int64_index_intermediate") must not be narrowed to i32 by the pass.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @int64_index_intermediate(
  tt.func public @int64_index_intermediate(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %big_stride: i64) {
    %pid = tt.get_program_id x : i32
    %pid_i64 = arith.extsi %pid : i32 to i64
    // COM: Intermediate computed at i64 specifically to avoid i32 overflow.
    %offset_i64 = arith.muli %pid_i64, %big_stride : i64
    %lane_range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %lane_range_i64 = arith.extsi %lane_range : tensor<1024xi32> to tensor<1024xi64>
    %offset_splat = tt.splat %offset_i64 : i64 -> tensor<1024xi64>
    %total_offset = arith.addi %offset_splat, %lane_range_i64 : tensor<1024xi64>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %total_offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    // COM: The offset tensor feeding the load/store must stay i64: narrowing
    // COM: it to i32 here would silently truncate the overflow-avoiding
    // COM: intermediate and produce a wrong address.
    // CHECK: tt.addptr {{.*}} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %val = tt.load %ptr : tensor<1024x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_ptr = tt.addptr %out_base, %total_offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    // CHECK: tt.addptr {{.*}} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    tt.store %out_ptr, %val : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// COM: Test 5 - Decomposing a `(x + C) * K` multiply into uniform/
// COM: non-uniform parts must not distribute into a bare `x * K` cross term.
// COM: Here `x` (a runtime tensor argument, standing in for a
// COM: divui/remui-derived index as PyTorch Inductor emits for its
// COM: cat/split fusions) ranges up to 63, and `x * 67108864` alone
// COM: overflows i32 (63 * 67108864 > INT32_MAX) even though the original
// COM: `(x + (-8)) * 67108864` does not for the intended (masked) range of
// COM: `x`. Widening the dangerous cross term to a wider type instead of
// COM: bailing out was tried and itself caused a separate correctness
// COM: regression, so the pass must bail out of distributing this multiply
// COM: entirely and keep the whole `(x + C) * K` expression fused, exactly
// COM: as originally written.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @mul_distribute_no_overflow(
  tt.func public @mul_distribute_no_overflow(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %x: tensor<64xi32>) {
    // CHECK: %[[C:.*]] = arith.constant dense<-8> : tensor<64xi32>
    %c-8_i32 = arith.constant dense<-8> : tensor<64xi32>
    // CHECK: %[[K:.*]] = arith.constant dense<67108864> : tensor<64xi32>
    %c67108864_i32 = arith.constant dense<67108864> : tensor<64xi32>
    // COM: The original fused `(x + C) * K` must survive unchanged -- not
    // COM: distributed into a standalone `x * K` cross term.
    // CHECK: %[[SHIFTED:.*]] = arith.addi %{{.*}}, %[[C]] : tensor<64xi32>
    %shifted = arith.addi %x, %c-8_i32 : tensor<64xi32>
    // CHECK: %[[OFFSET:.*]] = arith.muli %[[SHIFTED]], %[[K]] : tensor<64xi32>
    %offset = arith.muli %shifted, %c67108864_i32 : tensor<64xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %offset : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %val = tt.load %ptr : tensor<64x!tt.ptr<f32>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    tt.store %out_base, %val : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
