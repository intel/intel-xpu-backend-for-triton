// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s --check-prefix=NOFOLD
// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul="fold-trans=true" | FileCheck %s --check-prefix=FOLD

// COM: Attention backward dK/dV loop: p and ds are computed tiles whose transposes feed tt.dot as operand A.
// COM: With fold-trans they are computed transposed, pT = exp2(k q^T - m) and dsT = pT * (v do^T - Di): the
// COM: transposes of k and v cancel, and the only ones left are on the q and do descriptor loads.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // NOFOLD-LABEL: @attn_bwd_dkdv
  // NOFOLD-COUNT-2: tt.trans {{.*}} : tensor<64x64xf16
  // NOFOLD: scf.for
  // NOFOLD-COUNT-2: tt.trans {{.*}} : tensor<64x64xf32
  // NOFOLD-NOT: tt.trans

  // FOLD-LABEL: @attn_bwd_dkdv
  // FOLD-SAME: %[[K:[^:]+]]: tensor<64x64xf16, #{{.*}}>, %[[V:[^:]+]]: tensor<64x64xf16
  // FOLD-NOT: tt.trans
  // FOLD: scf.for
  // FOLD: %[[Q:.*]] = tt.descriptor_load
  // FOLD: %[[QT:.*]] = tt.trans %[[Q]]
  // FOLD: %[[DO:.*]] = tt.descriptor_load
  // FOLD: %[[DOT:.*]] = tt.trans %[[DO]]
  // FOLD-NOT: tt.trans
  // FOLD: %[[K_CVT:.*]] = ttg.convert_layout %[[K]]
  // FOLD: %[[QT_CVT:.*]] = ttg.convert_layout %[[QT]]
  // FOLD: %[[K_A:.*]] = ttg.convert_layout %[[K_CVT]]
  // FOLD: %[[QT_B:.*]] = ttg.convert_layout %[[QT_CVT]]
  // FOLD: tt.dot %[[K_A]], %[[QT_B]]
  // FOLD: tt.expand_dims {{.*}} {axis = 0 : i32}
  // FOLD: math.exp2
  // FOLD: tt.dot
  // FOLD: %[[V_CVT:.*]] = ttg.convert_layout %[[V]]
  // FOLD: %[[DOT_CVT:.*]] = ttg.convert_layout %[[DOT]]
  // FOLD: %[[V_A:.*]] = ttg.convert_layout %[[V_CVT]]
  // FOLD: %[[DOT_B:.*]] = ttg.convert_layout %[[DOT_CVT]]
  // FOLD: tt.dot %[[V_A]], %[[DOT_B]]
  // FOLD: tt.expand_dims {{.*}} {axis = 0 : i32}
  // FOLD: tt.dot
  // FOLD: scf.yield
  tt.func public @attn_bwd_dkdv(%q_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<64x64xf16>,
                                %k: tensor<64x64xf16, #blocked1>, %v: tensor<64x64xf16, #blocked1>,
                                %m: tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
                                %Di: tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
                                %num_steps: i32) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %vT = tt.trans %v {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %res:2 = scf.for %i = %c0_i32 to %num_steps step %c1_i32 iter_args(%dk = %cst, %dv = %cst) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>)  : i32 {
      %off = arith.muli %i, %c64_i32 : i32
      %q = tt.descriptor_load %q_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %do = tt.descriptor_load %do_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      // qk = q k^T, p = exp2(qk - m[:, None])
      %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %qk = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %m2 = tt.expand_dims %m {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %mb = tt.broadcast %m2 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %s = arith.subf %qk, %mb : tensor<64x64xf32, #blocked>
      %p = math.exp2 %s : tensor<64x64xf32, #blocked>
      // dv += p^T do
      %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %do_b = ttg.convert_layout %do : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dv_next = tt.dot %pT_a, %do_b, %dv, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      // dp = do v^T, ds = p * (dp - Di[:, None])
      %do_a = ttg.convert_layout %do : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %vT_b = ttg.convert_layout %vT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dp = tt.dot %do_a, %vT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %Di2 = tt.expand_dims %Di {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %Dib = tt.broadcast %Di2 : tensor<64x1xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %dpd = arith.subf %dp, %Dib : tensor<64x64xf32, #blocked>
      %ds = arith.mulf %p, %dpd : tensor<64x64xf32, #blocked>
      // dk += ds^T q
      %dsT = tt.trans %ds {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %dsT16 = arith.truncf %dsT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %dsT_a = ttg.convert_layout %dsT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %q_b = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dk_next = tt.dot %dsT_a, %q_b, %dk, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %dk_next, %dv_next : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
    }
    tt.return %res#0, %res#1 : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: Masked dV loop: the select condition is a leaf of the fold. The mask is built as written and transposed
// COM: once (as i32: extui, tt.trans, cmpi ne), its compares are not cloned in the transposed layout.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // NOFOLD-LABEL: @attn_bwd_dv_masked
  // NOFOLD: arith.cmpi sge, {{.*}} : tensor<64x64xi32, #{{.*}}>
  // NOFOLD-NOT: tt.trans {{.*}} : tensor<64x64xi1

  // FOLD-LABEL: @attn_bwd_dv_masked
  // FOLD: scf.for
  // FOLD: tt.dot
  // FOLD: %[[MASK:.*]] = arith.cmpi sge, {{.*}} : tensor<64x64xi32, #{{.*}}>
  // FOLD-NEXT: %[[EXT:.*]] = arith.extui %[[MASK]] : tensor<64x64xi1, #{{.*}}> to tensor<64x64xi32, #{{.*}}>
  // FOLD-NEXT: %[[TR:.*]] = tt.trans %[[EXT]] {{.*}} : tensor<64x64xi32, #{{.*}}> ->
  // FOLD: %[[MASKT:.*]] = arith.cmpi ne, %[[TR]], {{.*}} : tensor<64x64xi32, #{{.*}}>
  // FOLD-NOT: arith.cmpi
  // FOLD: arith.select %[[MASKT]]
  // FOLD: tt.dot
  // FOLD: scf.yield
  tt.func public @attn_bwd_dv_masked(%q_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<64x64xf16>,
                                     %k: tensor<64x64xf16, #blocked1>,
                                     %offs_n: tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>,
                                     %num_steps: i32) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %n2 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %nb = tt.broadcast %n2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %res = scf.for %i = %c0_i32 to %num_steps step %c1_i32 iter_args(%dv = %cst) -> (tensor<64x64xf32, #blocked>)  : i32 {
      %off = arith.muli %i, %c64_i32 : i32
      %q = tt.descriptor_load %q_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %do = tt.descriptor_load %do_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %qk = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      // causal mask: offs_m[:, None] >= offs_n[None, :]
      %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offs = tt.splat %off : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offs_m = arith.addi %offs, %range : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m2 = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %mb = tt.broadcast %m2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %mask = arith.cmpi sge, %mb, %nb : tensor<64x64xi32, #blocked>
      %p0 = math.exp2 %qk : tensor<64x64xf32, #blocked>
      %p = arith.select %mask, %p0, %cst : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %do_b = ttg.convert_layout %do : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dv_next = tt.dot %pT_a, %do_b, %dv, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %dv_next : tensor<64x64xf32, #blocked>
    }
    tt.return %res : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: A value of the region with a user outside it (%p is returned) would be needed in both orientations: no fold.
// COM: Same for a dot operand convert of the region shared with a dot outside it (%bT_b).
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @value_used_outside
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @value_used_outside(%desc: !tt.tensordesc<64x64xf16>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %s : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o, %p : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
  }

  // FOLD-LABEL: @operand_cvt_used_outside
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @operand_cvt_used_outside(%desc: !tt.tensordesc<64x64xf16>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %x = tt.descriptor_load %desc[%c192_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %s : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %x_a = ttg.convert_layout %x : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %o2 = tt.dot %x_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o, %o2 : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: A loop-carried accumulator in the slice is a block argument, which the fold does not handle: no fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @loop_carried_acc
  // FOLD: scf.for
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @loop_carried_acc(%desc: !tt.tensordesc<64x64xf16>, %num_steps: i32) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %res = scf.for %i = %c0_i32 to %num_steps step %c1_i32 iter_args(%acc = %cst) -> (tensor<64x64xf32, #blocked>)  : i32 {
      %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
      %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %s = tt.dot %a_a, %bT_b, %acc, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %p = math.exp2 %s : tensor<64x64xf32, #blocked>
      %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %o : tensor<64x64xf32, #blocked>
    }
    tt.return %res : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: An scf.if result in the slice is not an op the fold rewrites: no fold, although the slice has a dot.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @scf_if_in_slice
  // FOLD: %[[P:.*]] = arith.addf
  // FOLD: tt.trans %[[P]]
  tt.func public @scf_if_in_slice(%desc: !tt.tensordesc<64x64xf16>, %cond: i1) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<64x64xf32, #blocked>
    %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %w = scf.if %cond -> (tensor<64x64xf32, #blocked>) {
      scf.yield %cst : tensor<64x64xf32, #blocked>
    } else {
      scf.yield %cst1 : tensor<64x64xf32, #blocked>
    }
    %p0 = math.exp2 %s : tensor<64x64xf32, #blocked>
    %p = arith.addf %p0, %w : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: GEMM prologue dot(trans(x * s), y): without a dot in the slice the transpose would only move: no fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @gemm_prologue
  // FOLD: %[[XS:.*]] = arith.mulf
  // FOLD: tt.trans %[[XS]]
  tt.func public @gemm_prologue(%desc: !tt.tensordesc<64x64xf16>, %scale: f16) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %x = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %y = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %sc = tt.splat %scale : f16 -> tensor<64x64xf16, #blocked1>
    %xs = arith.mulf %x, %sc : tensor<64x64xf16, #blocked1>
    %xsT = tt.trans %xs {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %xsT_a = ttg.convert_layout %xsT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %y_b = ttg.convert_layout %y : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %xsT_a, %y_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: A computed tile defined before the loop (%e) is a leaf: it is transposed once, before the loop.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @outside_tile
  // FOLD: %[[E:.*]] = math.exp2
  // FOLD: tt.trans %[[E]]
  // FOLD: scf.for
  // FOLD-NOT: tt.trans {{.*}} : tensor<64x64xf32
  // FOLD: scf.yield
  tt.func public @outside_tile(%desc: !tt.tensordesc<64x64xf16>, %bias_desc: !tt.tensordesc<64x64xf32>, %num_steps: i32) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %e0 = tt.descriptor_load %bias_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked>
    %e = math.exp2 %e0 : tensor<64x64xf32, #blocked>
    %res = scf.for %i = %c0_i32 to %num_steps step %c1_i32 iter_args(%acc = %cst) -> (tensor<64x64xf32, #blocked>)  : i32 {
      %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
      %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %p = arith.addf %s, %e : tensor<64x64xf32, #blocked>
      %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %o = tt.dot %pT_a, %v_b, %acc, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %o : tensor<64x64xf32, #blocked>
    }
    tt.return %res : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: Chained transposes, a root (%pT) whose dot is in the slice of another root (%tT): %pT is reached as a
// COM: tt.trans leaf of the interior %p, which the rewrite would erase. No fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @chained_trans_of_computed
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  // FOLD: tt.trans
  tt.func public @chained_trans_of_computed(%q_desc: !tt.tensordesc<64x64xf16>, %k_desc: !tt.tensordesc<64x64xf16>,
                                            %v_desc: !tt.tensordesc<64x64xf16>, %w_desc: !tt.tensordesc<64x64xf16>) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %v_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %w = tt.descriptor_load %w_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %s : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %t = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %tT = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %tT16 = arith.truncf %tT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %tT_a = ttg.convert_layout %tT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %w_b = ttg.convert_layout %w : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %tT_a, %w_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: Chained transposes where the inner root (%lT) transposes a load: %lT is a tt.trans leaf of a load, not of
// COM: an interior value, so the region still folds: %t is computed transposed and no f32 tt.trans is left.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @chained_trans_of_load
  // FOLD-NOT: tt.trans {{.*}} : tensor<64x64xf32
  // FOLD: tt.return
  tt.func public @chained_trans_of_load(%desc: !tt.tensordesc<64x64xf16>) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %l = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %w = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %lT = tt.trans %l {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %lT_a = ttg.convert_layout %lT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %t = tt.dot %lT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %tT = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %tT16 = arith.truncf %tT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %tT_a = ttg.convert_layout %tT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %w_b = ttg.convert_layout %w : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %tT_a, %w_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: The mask is both a select condition and a value (uitofp). It is a leaf whichever use the walk reaches
// COM: first: built once as written and transposed as i32, its compare is not cloned in the transposed layout.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @mask_select_and_value
  // FOLD: scf.for
  // FOLD: %[[MASK:.*]] = arith.cmpi sge
  // FOLD-NEXT: arith.extui %[[MASK]]
  // FOLD-NOT: arith.cmpi sge
  // FOLD: arith.select
  // FOLD: arith.uitofp
  // FOLD: scf.yield
  tt.func public @mask_select_and_value(%q_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<64x64xf16>,
                                        %k: tensor<64x64xf16, #blocked1>,
                                        %offs_n: tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>,
                                        %num_steps: i32) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %n2 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %nb = tt.broadcast %n2 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %res = scf.for %i = %c0_i32 to %num_steps step %c1_i32 iter_args(%dv = %cst) -> (tensor<64x64xf32, #blocked>)  : i32 {
      %off = arith.muli %i, %c64_i32 : i32
      %q = tt.descriptor_load %q_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %do = tt.descriptor_load %do_desc[%off, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
      %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %qk = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offs = tt.splat %off : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offs_m = arith.addi %offs, %range : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m2 = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %mb = tt.broadcast %m2 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %mask = arith.cmpi sge, %mb, %nb : tensor<64x64xi32, #blocked>
      %p0 = math.exp2 %qk : tensor<64x64xf32, #blocked>
      %sel = arith.select %mask, %p0, %cst : tensor<64x64xi1, #blocked>, tensor<64x64xf32, #blocked>
      // COM: The walk pops the last operand first, so it reaches %mask through %u before the select.
      %u = arith.uitofp %mask : tensor<64x64xi1, #blocked> to tensor<64x64xf32, #blocked>
      %p = arith.addf %sel, %u : tensor<64x64xf32, #blocked>
      %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %do_b = ttg.convert_layout %do : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dv_next = tt.dot %pT_a, %do_b, %dv, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %dv_next : tensor<64x64xf32, #blocked>
    }
    tt.return %res : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: A function with a dot DPAS cannot run (fp32 inputPrecision = ieee) is left to the blocked lowering, so its
// COM: transposes are not folded, although the module supports DPAS.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @no_dpas_function
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @no_dpas_function(%desc: !tt.tensordesc<64x64xf16>, %desc32: !tt.tensordesc<64x64xf32>) -> (tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %s : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %x = tt.descriptor_load %desc32[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked1>
    %x_a = ttg.convert_layout %x : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %x_b = ttg.convert_layout %x : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %e = tt.dot %x_a, %x_b, %cst, inputPrecision = ieee : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o, %e : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: M = 8: qk is 8x64 and fits DPAS, the flipped 64x8 dot does not (N < executionSize). The consumer
// COM: still fits DPAS (tf32, K = 8). No fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @small_m
  // FOLD: tt.dot {{.*}} -> tensor<8x64xf32, #mma
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @small_m(%q_desc: !tt.tensordesc<8x64xf32>, %k_desc: !tt.tensordesc<64x64xf32>, %do_desc: !tt.tensordesc<8x64xf32>) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %cst8 = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf32> -> tensor<8x64xf32, #blocked1>
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked1>
    %do = tt.descriptor_load %do_desc[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf32> -> tensor<8x64xf32, #blocked1>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
    %q_a = ttg.convert_layout %q : tensor<8x64xf32, #blocked1> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %kT_b = ttg.convert_layout %kT : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %qk = tt.dot %q_a, %kT_b, %cst8, inputPrecision = tf32 : tensor<8x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x64xf32, #blocked>
    %p = math.exp2 %qk : tensor<8x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<8x64xf32, #blocked> -> tensor<64x8xf32, #blocked3>
    %pT_a = ttg.convert_layout %pT : tensor<64x8xf32, #blocked3> -> tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %do_b = ttg.convert_layout %do : tensor<8x64xf32, #blocked1> -> tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %dv = tt.dot %pT_a, %do_b, %cst, inputPrecision = tf32 : tensor<64x8xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %dv : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: The dot the transpose feeds is 64x8 (N < executionSize), so it stays off DPAS and its A operand gains
// COM: nothing from the fold, while qk fits DPAS flipped or not. No fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @small_consumer_n
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  // FOLD: tt.dot {{.*}} -> tensor<64x8xf32, #blocked>
  tt.func public @small_consumer_n(%q_desc: !tt.tensordesc<64x64xf16>, %k_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<64x8xf16>) -> tensor<64x8xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst8 = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %do = tt.descriptor_load %do_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x8xf16> -> tensor<64x8xf16, #blocked1>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %qk = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %qk : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %do_b = ttg.convert_layout %do : tensor<64x8xf16, #blocked1> -> tensor<64x8xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %dv = tt.dot %pT_a, %do_b, %cst8, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x8xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %dv : tensor<64x8xf32, #blocked>
  }
}

// -----

// COM: The roots are inside an scf.if: the computed tiles %p1 and %p2 of the enclosing block would become leaves,
// COM: each transposed before the scf.if whether or not the branch runs. No fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @roots_in_if
  // FOLD-NOT: tt.trans {{.*}} : tensor<64x64xf32
  // FOLD: scf.if
  // FOLD: arith.mulf
  // FOLD: %[[T:.*]] = arith.mulf
  // FOLD: tt.trans %[[T]]
  tt.func public @roots_in_if(%q_desc: !tt.tensordesc<64x64xf16>, %k_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<64x64xf16>, %s_desc: !tt.tensordesc<64x64xf32>, %c: i1) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %do = tt.descriptor_load %do_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %s1 = tt.descriptor_load %s_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked>
    %s2 = tt.descriptor_load %s_desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf32> -> tensor<64x64xf32, #blocked>
    %p1 = math.exp2 %s1 : tensor<64x64xf32, #blocked>
    %p2 = math.exp2 %s2 : tensor<64x64xf32, #blocked>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %r = scf.if %c -> (tensor<64x64xf32, #blocked>) {
      %q_a = ttg.convert_layout %q : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %qk = tt.dot %q_a, %kT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      %t1 = arith.mulf %qk, %p1 : tensor<64x64xf32, #blocked>
      %t = arith.mulf %t1, %p2 : tensor<64x64xf32, #blocked>
      %tT = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
      %tT16 = arith.truncf %tT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
      %tT_a = ttg.convert_layout %tT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %do_b = ttg.convert_layout %do : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %dv = tt.dot %tT_a, %do_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
      scf.yield %dv : tensor<64x64xf32, #blocked>
    } else {
      scf.yield %cst : tensor<64x64xf32, #blocked>
    }
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: A 32x64 q tile, M != N: the flipped dot is 64x32 and the row vector m (32 values, one per q row) is
// COM: expanded along the other axis, 1x32, and broadcast to 64x32.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @nonsquare
  // FOLD-NOT: tt.trans {{.*}} : tensor<32x64xf32
  // FOLD: tt.dot {{.*}} -> tensor<64x32xf32, #mma
  // FOLD: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<32xf32, {{.*}}> -> tensor<1x32xf32
  // FOLD: tt.broadcast {{.*}} -> tensor<64x32xf32
  // FOLD: math.exp2 {{.*}} : tensor<64x32xf32
  // FOLD-NOT: tt.trans {{.*}}xf32
  // FOLD: tt.return
  tt.func public @nonsquare(%q_desc: !tt.tensordesc<32x64xf16>, %k_desc: !tt.tensordesc<64x64xf16>, %do_desc: !tt.tensordesc<32x64xf16>,
                            %m: tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %cst32 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %q = tt.descriptor_load %q_desc[%c0_i32, %c0_i32] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16, #blocked1>
    %k = tt.descriptor_load %k_desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %do = tt.descriptor_load %do_desc[%c0_i32, %c0_i32] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16, #blocked1>
    %kT = tt.trans %k {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %q_a = ttg.convert_layout %q : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %kT_b = ttg.convert_layout %kT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %qk = tt.dot %q_a, %kT_b, %cst32, inputPrecision = tf32 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
    %m2 = tt.expand_dims %m {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xf32, #blocked>
    %mb = tt.broadcast %m2 : tensor<32x1xf32, #blocked> -> tensor<32x64xf32, #blocked>
    %s = arith.subf %qk, %mb : tensor<32x64xf32, #blocked>
    %p = math.exp2 %s : tensor<32x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<32x64xf32, #blocked> -> tensor<64x32xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x32xf32, #blocked3> to tensor<64x32xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x32xf16, #blocked3> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %do_b = ttg.convert_layout %do : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %dv = tt.dot %pT_a, %do_b, %cst, inputPrecision = tf32 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %dv : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: Without 2D block IO a descriptor load leaf is not a cheap column-major load. No fold.
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 1], warpsPerCTA = [1, 8], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate} {
  // FOLD-LABEL: @no_2d_block_io
  // FOLD: %[[P:.*]] = math.exp2
  // FOLD: tt.trans %[[P]]
  tt.func public @no_2d_block_io(%desc: !tt.tensordesc<64x64xf16>) -> tensor<64x64xf32, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %a = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %b = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %v = tt.descriptor_load %desc[%c128_i32, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked1>
    %bT = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #blocked2>
    %a_a = ttg.convert_layout %a : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bT_b = ttg.convert_layout %bT : tensor<64x64xf16, #blocked2> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %s = tt.dot %a_a, %bT_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %p = math.exp2 %s : tensor<64x64xf32, #blocked>
    %pT = tt.trans %p {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    %pT16 = arith.truncf %pT : tensor<64x64xf32, #blocked3> to tensor<64x64xf16, #blocked3>
    %pT_a = ttg.convert_layout %pT16 : tensor<64x64xf16, #blocked3> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %v_b = ttg.convert_layout %v : tensor<64x64xf16, #blocked1> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %o = tt.dot %pT_a, %v_b, %cst, inputPrecision = tf32 : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}
