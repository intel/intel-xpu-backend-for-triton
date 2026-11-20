// RUN: triton-opt %s --allow-unregistered-dialect --mlir-disable-threading -triton-intel-narrow-broadcast-loads="num-warps=16 threads-per-warp=32" | FileCheck %s
// RUN: triton-opt %s --allow-unregistered-dialect --mlir-disable-threading -triton-intel-narrow-broadcast-loads="num-warps=16 threads-per-warp=32" -canonicalize | FileCheck %s --check-prefix=FOLD

// A row-broadcast load in a dot epilogue is loaded once as 1xN under (column mask & some-row-valid) and
// broadcast; a select with the original mask and `other` keeps every value. Negatives keep the wide load.

module {
  // CHECK-LABEL: tt.func @r1_issue8813
  // CHECK: %[[B:.*]] = arith.muli
  // CHECK: %[[MN2:.*]] = tt.expand_dims %{{.*}} {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
  // CHECK: %[[MASK:.*]] = arith.andi %{{.*}}, %{{.*}} : tensor<64x128xi1>
  // CHECK: %[[P1:.*]] = tt.addptr %{{.*}} : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
  // CHECK: %[[OTHER:.*]] = arith.constant dense<0.000000e+00> : tensor<64x128xf16>
  // CHECK-DAG: %[[BOUND:.*]] = arith.constant 8448 : i32
  // CHECK-DAG: %[[THR:.*]] = arith.constant 2147483584 : i32
  // CHECK-DAG: %[[MIN:.*]] = arith.constant -2147483648 : i32
  // CHECK: %[[LT:.*]] = arith.cmpi slt, %[[B]], %[[BOUND]] : i32
  // CHECK: %[[WR:.*]] = arith.cmpi sgt, %[[B]], %[[THR]] : i32
  // CHECK: %[[NM:.*]] = arith.cmpi sgt, %[[BOUND]], %[[MIN]] : i32
  // CHECK: %[[W:.*]] = arith.andi %[[WR]], %[[NM]] : i1
  // CHECK: %[[ANY:.*]] = arith.ori %[[LT]], %[[W]] : i1
  // CHECK: %[[ANYS:.*]] = tt.splat %[[ANY]] : i1 -> tensor<1x128xi1>
  // CHECK: %[[Q:.*]] = arith.andi %[[MN2]], %[[ANYS]] : tensor<1x128xi1>
  // CHECK: %[[NZ:.*]] = arith.constant dense<0.000000e+00> : tensor<1x128xf16>
  // CHECK: %[[V:.*]] = tt.load %[[P1]], %[[Q]], %[[NZ]] {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>} : tensor<1x128x!tt.ptr<f16>>
  // CHECK: %[[BC:.*]] = tt.broadcast %[[V]] : tensor<1x128xf16> -> tensor<64x128xf16>
  // CHECK: %[[SEL:.*]] = arith.select %[[MASK]], %[[BC]], %[[OTHER]] : tensor<64x128xi1>, tensor<64x128xf16>
  // CHECK-NOT: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  // CHECK: arith.extf %[[SEL]]
  tt.func @r1_issue8813(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %A: tensor<64x64x!tt.ptr<f16>>) -> tensor<64x128xf32> {
    %c0 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : i32
    %c64k = arith.constant 64 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = scf.for %k = %c0 to %c1024 step %c64k iter_args(%it = %acc0) -> (tensor<64x128xf32>)  : i32 {
      %ta = tt.load %A : tensor<64x64x!tt.ptr<f16>>
      %d = tt.dot %ta, %bm, %it : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
      scf.yield %d : tensor<64x128xf32>
    }
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x128xf16>
    %bias = tt.load %bp, %mask, %zero {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>} : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r2_inductor
  // CHECK: %[[M:.*]] = tt.splat %arg3 : i32 -> tensor<64x1xi32>
  // CHECK: arith.cmpi slt, %{{.*}}, %arg3 : i32
  // CHECK: %[[ANY:.*]] = arith.ori
  // CHECK: %[[Q:.*]] = tt.splat %[[ANY]] : i1 -> tensor<1x128xi1>
  // CHECK: %[[V:.*]] = tt.load %{{.*}}, %[[Q]], %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK: %[[Z:.*]] = arith.constant dense<0.000000e+00> : tensor<64x128xf16>
  // CHECK: arith.select %{{.*}}, %{{.*}}, %[[Z]] : tensor<64x128xi1>, tensor<64x128xf16>
  tt.func @r2_inductor(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %M: i32) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %im = tt.expand_dims %rm {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %Ms = tt.splat %M : i32 -> tensor<64x1xi32>
    %mm = arith.cmpi slt, %im, %Ms : tensor<64x1xi32>
    %mask = tt.broadcast %mm : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r3_other_policy_hints
  // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_first>} : tensor<1x128x!tt.ptr<f16>>
  // CHECK: arith.select %{{.*}}, %{{.*}}, %arg3 {tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[1, 8]> : tensor<2xi32>} : tensor<64x128xi1>, tensor<64x128xf16>
  tt.func @r3_other_policy_hints(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %O: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask, %O {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_first>, tt.constancy = dense<[1, 1]> : tensor<2xi32>, tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[1, 8]> : tensor<2xi32>} : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r4_two_uses
  // CHECK: %[[SEL:.*]] = arith.select
  // CHECK: tt.store %{{.*}}, %[[SEL]], %{{.*}} : tensor<64x128x!tt.ptr<f16>>
  // CHECK: arith.extf %[[SEL]]
  tt.func @r4_two_uses(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %Q: tensor<64x128x!tt.ptr<f16>>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    tt.store %Q, %bias, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r5_c_leaves
  // CHECK-DAG: %[[F:.*]] = tt.splat %arg3 : i1 -> tensor<1x128xi1>
  // CHECK-DAG: %[[T:.*]] = arith.constant dense<true> : tensor<1x128xi1>
  // CHECK: tt.load %{{.*}}, %[[Q:.*]], %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  tt.func @r5_c_leaves(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %flag: i1) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %ge = arith.constant dense<0> : tensor<128xi32>
    %mg = arith.cmpi sge, %rn, %ge : tensor<128xi32>
    %mg2 = tt.expand_dims %mg {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mgb = tt.broadcast %mg2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %fs = tt.splat %flag : i1 -> tensor<64x128xi1>
    %tru = arith.constant dense<true> : tensor<64x128xi1>
    %m1 = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %m2 = arith.andi %m1, %mgb : tensor<64x128xi1>
    %m3 = arith.andi %fs, %m2 : tensor<64x128xi1>
    %mask = arith.andi %m3, %tru : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r6_swapped
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NOT: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @r6_swapped(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %r, %bs : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi sgt, %mb, %rm : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r8_dot_scaled
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NOT: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @r8_dot_scaled(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %sa: tensor<64x64xf8E5M2>, %sas: tensor<64x2xi8>, %sb: tensor<64x128xf8E5M2>, %sbs: tensor<128x2xi8>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot_scaled %sa scale %sas, %sb scale %sbs, %acc0 lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<64x64xf8E5M2>, tensor<64x2xi8> * tensor<64x128xf8E5M2>, tensor<128x2xi8> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @r9_assume_before_load
  // CHECK: llvm.intr.assume
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NOT: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @r9_assume_before_load(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %tr = arith.constant true
    llvm.intr.assume %tr : i1
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @w1_no_wrap_false
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD-LABEL: tt.func @w1_no_wrap_false
  // FOLD-NOT: tt.load
  // FOLD: tt.return
  tt.func @w1_no_wrap_false(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %b = arith.constant 2147483584 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<0> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mmb : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @w2_wrap_min
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD-LABEL: tt.func @w2_wrap_min
  // FOLD-NOT: tt.load
  // FOLD: tt.return
  tt.func @w2_wrap_min(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %b = arith.constant 2147483585 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<-2147483648> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mmb : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @w3_wrap_min_plus1
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD-LABEL: tt.func @w3_wrap_min_plus1
  // FOLD: tt.load %{{[a-z0-9_]+}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD: tt.return
  tt.func @w3_wrap_min_plus1(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %b = arith.constant 2147483585 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<-2147483647> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mmb : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @w4_ordinary
  // CHECK: tt.load {{.*}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD-LABEL: tt.func @w4_ordinary
  // FOLD: tt.load %{{[a-z0-9_]+}} : tensor<1x128x!tt.ptr<f16>>
  // FOLD: tt.return
  tt.func @w4_ordinary(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %b = arith.constant 0 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mmb : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n1_volatile
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n1_volatile(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask {isVolatile = true} : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n2_two_row_leaves
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n2_two_row_leaves(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %b2: i32) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %bs2 = tt.splat %b2 : i32 -> tensor<64xi32>
    %rm2 = arith.addi %bs2, %r : tensor<64xi32>
    %mbx = arith.constant dense<8448> : tensor<64xi32>
    %mmx = arith.cmpi slt, %rm2, %mbx : tensor<64xi32>
    %mm2x = tt.expand_dims %mmx {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb2 = tt.broadcast %mm2x : tensor<64x1xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mmb2 : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n2b_unsupported_pred
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n2b_unsupported_pred(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi sge, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n3_ult
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n3_ult(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi ult, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n4_too_wide
  // CHECK: tt.load {{.*}} : tensor<64x1024x!tt.ptr<f16>>
  tt.func @n4_too_wide(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x1024xf16>) -> tensor<64x1024xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x1024xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x1024xf16> -> tensor<64x1024xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x1024xi1>
    %rn = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %nb = arith.constant dense<512> : tensor<1024xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<1024xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<1024xi1> -> tensor<1x1024xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x1024xi1> -> tensor<64x1024xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x1024xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<1024xi32> -> tensor<1x1024xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x1024x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x1024x!tt.ptr<f16>>, tensor<1x1024xi32>
    %bp = tt.broadcast %p1 : tensor<1x1024x!tt.ptr<f16>> -> tensor<64x1024x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x1024x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x1024xf16> to tensor<64x1024xf32>
    %t = arith.addf %acc, %bf : tensor<64x1024xf32>
    tt.return %t : tensor<64x1024xf32>
  }

  // CHECK-LABEL: tt.func @n5_no_dot
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n5_no_dot(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    tt.return %bf : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n6_full_2d_ptr
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n6_full_2d_ptr(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %o2 = tt.broadcast %rn2 : tensor<1x128xi32> -> tensor<64x128xi32>
    %pw = tt.splat %P : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>>
    %p2 = tt.addptr %pw, %o2 : tensor<64x128x!tt.ptr<f16>>, tensor<64x128xi32>
    %bias = tt.load %p2, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n7_unmasked
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n7_unmasked(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n8_prior_store
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n8_prior_store(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %Q: tensor<64x128x!tt.ptr<f16>>) -> tensor<64x128xf32> {
    tt.store %Q, %bm : tensor<64x128x!tt.ptr<f16>>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n9_nested_prior_store
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n9_nested_prior_store(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %Q: tensor<64x128x!tt.ptr<f16>>, %c: i1) -> tensor<64x128xf32> {
    scf.if %c {
      tt.store %Q, %bm : tensor<64x128x!tt.ptr<f16>>
    }
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n11_prior_atomic
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n11_prior_atomic(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %Q: tensor<64x128x!tt.ptr<f16>>) -> tensor<64x128xf32> {
    %am = arith.constant dense<true> : tensor<64x128xi1>
    %at = tt.atomic_rmw fadd, relaxed, gpu, %Q, %bm, %am : (tensor<64x128x!tt.ptr<f16>>, tensor<64x128xf16>, tensor<64x128xi1>) -> tensor<64x128xf16>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n12_impure_asm
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n12_impure_asm(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %ia = tt.elementwise_inline_asm "mov.u32 $0, 0;" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n13_unknown_attr
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n13_unknown_attr(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask {foo.bar = 1 : i32} : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n14_prior_call
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n14_prior_call(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    tt.call @callee() : () -> ()
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n15_unrelated_dot
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n15_unrelated_dot(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %x: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %x, %bf : tensor<64x128xf32>
    tt.return %acc : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n17_opaque_region
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n17_opaque_region(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    "test.opaque"() ({
      %z = arith.constant 0 : i32
    }) : () -> ()
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n18_range_start
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n18_range_start(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 72 : i32, start = 8 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  // CHECK-LABEL: tt.func @n10_loop_store
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<f16>>
  tt.func @n10_loop_store(%P: !tt.ptr<f16>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>, %Q: tensor<64x128x!tt.ptr<f16>>) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1  : i32 {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<f16> -> tensor<1x128x!tt.ptr<f16>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<f16>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<f16>> -> tensor<64x128x!tt.ptr<f16>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<f16>>
    %bf = arith.extf %bias : tensor<64x128xf16> to tensor<64x128xf32>
    %t = arith.addf %acc, %bf : tensor<64x128xf32>
    %th = arith.truncf %t : tensor<64x128xf32> to tensor<64x128xf16>
    tt.store %Q, %th : tensor<64x128x!tt.ptr<f16>>
    }
    tt.return
  }

  // CHECK-LABEL: tt.func @n16_fp4
  // CHECK: tt.load {{.*}} : tensor<64x128x!tt.ptr<!tt.fp4e2m1x2>>
  tt.func @n16_fp4(%P: !tt.ptr<!tt.fp4e2m1x2>, %a: tensor<64x64xf16>, %bm: tensor<64x128xf16>) -> tensor<64x128xf32> {
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %acc = tt.dot %a, %bm, %acc0 : tensor<64x64xf16> * tensor<64x128xf16> -> tensor<64x128xf32>
    %pid = tt.get_program_id x : i32
    %c64 = arith.constant 64 : i32
    %b = arith.muli %pid, %c64 : i32
    %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %bs = tt.splat %b : i32 -> tensor<64xi32>
    %rm = arith.addi %bs, %r : tensor<64xi32>
    %mb = arith.constant dense<8448> : tensor<64xi32>
    %mm = arith.cmpi slt, %rm, %mb : tensor<64xi32>
    %mm2 = tt.expand_dims %mm {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %mmb = tt.broadcast %mm2 : tensor<64x1xi1> -> tensor<64x128xi1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %nb = arith.constant dense<512> : tensor<128xi32>
    %mn = arith.cmpi slt, %rn, %nb : tensor<128xi32>
    %mn2 = tt.expand_dims %mn {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %mnb = tt.broadcast %mn2 : tensor<1x128xi1> -> tensor<64x128xi1>
    %mask = arith.andi %mmb, %mnb : tensor<64x128xi1>
    %rn2 = tt.expand_dims %rn {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %ps = tt.splat %P : !tt.ptr<!tt.fp4e2m1x2> -> tensor<1x128x!tt.ptr<!tt.fp4e2m1x2>>
    %p1 = tt.addptr %ps, %rn2 : tensor<1x128x!tt.ptr<!tt.fp4e2m1x2>>, tensor<1x128xi32>
    %bp = tt.broadcast %p1 : tensor<1x128x!tt.ptr<!tt.fp4e2m1x2>> -> tensor<64x128x!tt.ptr<!tt.fp4e2m1x2>>
    %bias = tt.load %bp, %mask : tensor<64x128x!tt.ptr<!tt.fp4e2m1x2>>
    %t = tt.elementwise_inline_asm "mov.u32 $0, $1;" {constraints = "=r,r,r", packed_element = 1 : i32, pure = true} %bias, %acc : tensor<64x128x!tt.fp4e2m1x2>, tensor<64x128xf32> -> tensor<64x128xf32>
    tt.return %t : tensor<64x128xf32>
  }

  tt.func private @callee()
}
