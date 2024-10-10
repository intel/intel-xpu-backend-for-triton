// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK-DAG: llvm.func spir_funccc @_Z16get_sub_group_idv() -> i32 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["nosync"], will_return}

llvm.func @gen_special_regs() -> i32 {
  // CHECK-LABEL: gen_special_regs

  // CHECK: llvm.call spir_funccc @_Z16get_sub_group_idv() {{.*}} : () -> i32
  %1 = triton_gen.subgroup.id : i32

  // CHECK: llvm.call spir_funccc @_Z22get_sub_group_local_idv() {{.*}} : () -> i32
  %14 = triton_gen.subgroup.local.id : i32

  llvm.return %1 : i32
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z31intel_work_group_barrier_arriveii(i32, i32) attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z29intel_work_group_barrier_waitii(i32, i32) attributes {convergent, no_unwind, will_return}

llvm.func @triton_gen.split_barrier() {
  // CHECK-LABEL: triton_gen.split_barrier() {
  // CHECK-DAG: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:     llvm.call spir_funccc @_Z31intel_work_group_barrier_arriveii([[ZERO]], [[ONE]]) {{.*}} : (i32, i32) -> ()
  // CHECK-DAG: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:     llvm.call spir_funccc @_Z29intel_work_group_barrier_waitii([[ZERO]], [[ONE]]) {{.*}} : (i32, i32) -> ()
  triton_gen.split_barrier_signal {mem_fence=None, mem_scope=WorkGroup}
  triton_gen.split_barrier_wait {mem_fence=None, mem_scope=WorkGroup}
  llvm.return
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_addij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_mulij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_maxij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_minij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_andij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z29sub_group_clustered_reduce_orij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z30sub_group_clustered_reduce_xorij(i32, i32) -> i32 attributes {convergent, no_unwind, will_return}

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 32>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_addij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %1 = triton_gen.sub_group_reduce add %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_mulij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %2 = triton_gen.sub_group_reduce mul %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_minij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %3 = triton_gen.sub_group_reduce min %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_maxij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %4 = triton_gen.sub_group_reduce max %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_andij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %5 = triton_gen.sub_group_reduce and %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z29sub_group_clustered_reduce_orij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %6 = triton_gen.sub_group_reduce or %0 {size = 16} : i32
    // CHECK: [[SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z30sub_group_clustered_reduce_xorij([[VAL]], [[SIZE]]) {{.*}} : (i32, i32) -> i32
    %7 = triton_gen.sub_group_reduce xor %0 {size = 16} : i32
    llvm.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_addi(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_muli(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_mini(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_maxi(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_andi(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z31sub_group_non_uniform_reduce_ori(i32) -> i32
// CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_xori(i32) -> i32

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_addi([[VAL]]) {{.*}} : (i32) -> i32
    %1 = triton_gen.sub_group_reduce add %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_muli([[VAL]]) {{.*}} : (i32) -> i32
    %2 = triton_gen.sub_group_reduce mul %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_mini([[VAL]]) {{.*}} : (i32) -> i32
    %3 = triton_gen.sub_group_reduce min %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_maxi([[VAL]]) {{.*}} : (i32) -> i32
    %4 = triton_gen.sub_group_reduce max %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_andi([[VAL]]) {{.*}} : (i32) -> i32
    %5 = triton_gen.sub_group_reduce and %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z31sub_group_non_uniform_reduce_ori([[VAL]]) {{.*}} : (i32) -> i32
    %6 = triton_gen.sub_group_reduce or %0 {size = 16} : i32
    // CHECK: llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_xori([[VAL]]) {{.*}} : (i32) -> i32
    %7 = triton_gen.sub_group_reduce xor %0 {size = 16} : i32
    llvm.return
  }
}

// -----

// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_addi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_muli(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_maxi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_mini(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_andi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z39sub_group_non_uniform_scan_exclusive_ori(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_xori(i32) -> i32 attributes {convergent, no_unwind, will_return}

// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_addi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_muli(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_maxi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_mini(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_andi(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z39sub_group_non_uniform_scan_inclusive_ori(i32) -> i32 attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_xori(i32) -> i32 attributes {convergent, no_unwind, will_return}

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  llvm.func @triton_gen.sub_group_scan() {
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_addi([[VAL]]) {{.*}} : (i32) -> i32
    %1 = triton_gen.sub_group_scan add %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_muli([[VAL]]) {{.*}} : (i32) -> i32
    %2 = triton_gen.sub_group_scan mul %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_mini([[VAL]]) {{.*}} : (i32) -> i32
    %3 = triton_gen.sub_group_scan min %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_maxi([[VAL]]) {{.*}} : (i32) -> i32
    %4 = triton_gen.sub_group_scan max %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_andi([[VAL]]) {{.*}} : (i32) -> i32
    %5 = triton_gen.sub_group_scan and %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z39sub_group_non_uniform_scan_exclusive_ori([[VAL]]) {{.*}} : (i32) -> i32
    %6 = triton_gen.sub_group_scan or %0 {kind = exclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_exclusive_xori([[VAL]]) {{.*}} : (i32) -> i32
    %7 = triton_gen.sub_group_scan xor %0 {kind = exclusive} : i32

    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_addi([[VAL]]) {{.*}} : (i32) -> i32
    %8 = triton_gen.sub_group_scan add %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_muli([[VAL]]) {{.*}} : (i32) -> i32
    %9 = triton_gen.sub_group_scan mul %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_mini([[VAL]]) {{.*}} : (i32) -> i32
    %10 = triton_gen.sub_group_scan min %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_maxi([[VAL]]) {{.*}} : (i32) -> i32
    %11 = triton_gen.sub_group_scan max %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_andi([[VAL]]) {{.*}} : (i32) -> i32
    %12 = triton_gen.sub_group_scan and %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z39sub_group_non_uniform_scan_inclusive_ori([[VAL]]) {{.*}} : (i32) -> i32
    %13 = triton_gen.sub_group_scan or %0 {kind = inclusive} : i32
    // CHECK: llvm.call spir_funccc @_Z40sub_group_non_uniform_scan_inclusive_xori([[VAL]]) {{.*}} : (i32) -> i32
    %14 = triton_gen.sub_group_scan xor %0 {kind = inclusive} : i32

    llvm.return
  }
}

// -----

// CHECK: llvm.func spir_funccc @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.i8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.i8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z36intel_sub_group_i8_i8_matrix_mad_k32Dv8_sDv8_iS0_(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z36intel_sub_group_u8_u8_matrix_mad_k32Dv8_sDv8_iS0_(vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.u8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.u8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z36intel_sub_group_u8_u8_matrix_mad_k32Dv8_sDv8_iS0_(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xi32>) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa = u8, pb = u8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.bf16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.bf16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iDv8_f(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(%arg1, %arg2, %arg0) {{.*}} : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_(vector<4xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f32(%c : vector<8xf32>, %a : vector<4xf32>, %b : vector<8xf32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f32(%arg0: vector<8xf32>, %arg1: vector<4xf32>, %arg2: vector<8xf32>) {
  // CHECK-NEXT: llvm.call spir_funccc @_Z39intel_sub_group_tf32_tf32_matrix_mad_k8Dv4_fDv8_fS0_
  // CHECK-SAME:    (%arg1, %arg2, %arg0)
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<4xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh(vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f16_accum(%c: vector<8xf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f16_accum(%arg0: vector<8xf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: lvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_Dh
  // CHECK-SAME:    (%arg1, %arg2, %arg0)
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<8xi16>, vector<8xi32>, vector<8xf16>) -> vector<8xf16>
  %0 = triton_gen.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf16>, vector<8xi16>, vector<8xi32>) -> vector<8xf16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iS_(vector<8xi16>, vector<8xi32>, vector<8xi16>) -> vector<8xi16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.bf16_accum(%c: vector<8xbf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.bf16_accum(%arg0: vector<8xbf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: [[CAST:%.*]] = llvm.bitcast %arg0 : vector<8xbf16> to vector<8xi16>
  // CHECK-NEXT: [[RES:%.*]] = llvm.call spir_funccc @_Z40intel_sub_group_bf16_bf16_matrix_mad_k16Dv8_sDv8_iS_
  // CHECK-SAME:    (%arg1, %arg2, [[CAST]])
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (vector<8xi16>, vector<8xi32>, vector<8xi16>) -> vector<8xi16>
  %0 = triton_gen.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xbf16>, vector<8xi16>, vector<8xi32>) -> vector<8xbf16>
  // CHECK-NEXT: {{%.*}} = llvm.bitcast [[RES]] : vector<8xi16> to vector<8xbf16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z30intel_sub_group_block_read_us2PU3AS3t(!llvm.ptr<3>) -> vector<2xi16> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.simdblockread(%ptr: !llvm.ptr<3>) {
  // CHECK:     llvm.func @triton_gen.simdblockread(%arg0: !llvm.ptr<3>) {
  // CHECK:       llvm.call spir_funccc @_Z30intel_sub_group_block_read_us2PU3AS3t(%arg0) {{.*}} : (!llvm.ptr<3>) -> vector<2xi16>
  %ret = triton_gen.simdblockread %ptr : (!llvm.ptr<3>) -> vector<2xi16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z31intel_sub_group_block_write_us2PU3AS3tDv2_t(!llvm.ptr<3>, vector<2xi16>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.simdblockwrite(%ptr: !llvm.ptr<3>, %val : vector<2xi16>) {
  // CHECK:     llvm.func @triton_gen.simdblockwrite(%arg0: !llvm.ptr<3>, %arg1: vector<2xi16>) {
  // CHECK:       llvm.call spir_funccc @_Z31intel_sub_group_block_write_us2PU3AS3tDv2_t(%arg0, %arg1) {{.*}} : (!llvm.ptr<3>, vector<2xi16>) -> ()
  triton_gen.simdblockwrite %ptr, %val : (!llvm.ptr<3>, vector<2xi16>)
  llvm.return
}
