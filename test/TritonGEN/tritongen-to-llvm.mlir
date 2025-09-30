// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.i8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.i8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-DAG:  [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG:  [[C51:%.*]] = llvm.mlir.constant(51 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i([[C32]], %arg1, %arg2, %arg0, [[C51]]) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i(i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.u8(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.u8(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-DAG:  [[C32:%.*]] = llvm.mlir.constant(32 : i32) : i32
  // CHECK-DAG:  [[C48:%.*]] = llvm.mlir.constant(48 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS0_i([[C32]], %arg1, %arg2, %arg0, [[C48]]) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xi32>, i32) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa = u8, pb = u8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.bf16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.bf16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-DAG:  [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  [[C12288:%.*]] = llvm.mlir.constant(12288 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi([[C16]], %arg1, %arg2, %arg0, [[C12288]]) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f16(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f16(%arg0: vector<8xf32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-DAG:  [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  [[C3072:%.*]] = llvm.mlir.constant(3072 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi([[C16]], %arg1, %arg2, %arg0, [[C3072]]) {{.*}} : (i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS0_i(i32, vector<4xf32>, vector<8xf32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f32(%c : vector<8xf32>, %a : vector<4xf32>, %b : vector<8xf32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f32(%arg0: vector<8xf32>, %arg1: vector<4xf32>, %arg2: vector<8xf32>) {
  // CHECK-DAG:  [[C8:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  [[C768:%.*]] = llvm.mlir.constant(768 : i32) : i32
  // CHECK-NEXT: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv4_fDv8_fS0_i
  // CHECK-SAME:    ([[C8]], %arg1, %arg2, %arg0, [[C768]])
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (i32, vector<4xf32>, vector<8xf32>, vector<8xf32>, i32) -> vector<8xf32>
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi(i32, vector<8xi16>, vector<8xi32>, vector<8xf16>, i32) -> vector<8xf16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.f16_accum(%c: vector<8xf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.f16_accum(%arg0: vector<8xf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-DAG:  [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  [[C3072:%.*]] = llvm.mlir.constant(3072 : i32) : i32
  // CHECK-NEXT: lvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_Dhi
  // CHECK-SAME:    ([[C16]], %arg1, %arg2, %arg0, [[C3072]])
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (i32, vector<8xi16>, vector<8xi32>, vector<8xf16>, i32) -> vector<8xf16>
  %0 = triton_gen.dpas %c, %a, %b {pa = f16, pb = f16, rc = 8} : (vector<8xf16>, vector<8xi16>, vector<8xi32>) -> vector<8xf16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i(i32, vector<8xi16>, vector<8xi32>, vector<8xi16>, i32) -> vector<8xi16> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.dpas.bf16_accum(%c: vector<8xbf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:     llvm.func @triton_gen.dpas.bf16_accum(%arg0: vector<8xbf16>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT: [[CAST:%.*]] = llvm.bitcast %arg0 : vector<8xbf16> to vector<8xi16>
  // CHECK-DAG:  [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  [[C12300:%.*]] = llvm.mlir.constant(12300 : i32) : i32
  // CHECK-NEXT: [[RES:%.*]] = llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iS_i
  // CHECK-SAME:    ([[C16]], %arg1, %arg2, [[CAST]], [[C12300]])
  // CHECK-SAME:    convergent
  // CHECK-SAME:    : (i32, vector<8xi16>, vector<8xi32>, vector<8xi16>, i32) -> vector<8xi16>
  %0 = triton_gen.dpas %c, %a, %b {pa = bf16, pb = bf16, rc = 8} : (vector<8xbf16>, vector<8xi16>, vector<8xi32>) -> vector<8xbf16>
  // CHECK-NEXT: {{%.*}} = llvm.bitcast [[RES]] : vector<8xi16> to vector<8xbf16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z30intel_sub_group_block_read_us2PU3AS3t(!llvm.ptr<3>) -> vector<2xi16> attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.sub_group_block_read(%ptr: !llvm.ptr<3>) {
  // CHECK:     llvm.func @triton_gen.sub_group_block_read(%arg0: !llvm.ptr<3>) {
  // CHECK:       llvm.call spir_funccc @_Z30intel_sub_group_block_read_us2PU3AS3t(%arg0) {{.*}} : (!llvm.ptr<3>) -> vector<2xi16>
  %ret = triton_gen.sub_group_block_read %ptr : !llvm.ptr<3> -> vector<2xi16>
  llvm.return
}

// -----

// CHECK: llvm.func spir_funccc @_Z31intel_sub_group_block_write_us2PU3AS3tDv2_t(!llvm.ptr<3>, vector<2xi16>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, will_return}

llvm.func @triton_gen.sub_group_block_write(%ptr: !llvm.ptr<3>, %val : vector<2xi16>) {
  // CHECK:     llvm.func @triton_gen.sub_group_block_write(%arg0: !llvm.ptr<3>, %arg1: vector<2xi16>) {
  // CHECK:       llvm.call spir_funccc @_Z31intel_sub_group_block_write_us2PU3AS3tDv2_t(%arg0, %arg1) {{.*}} : (!llvm.ptr<3>, vector<2xi16>) -> ()
  triton_gen.sub_group_block_write %ptr, %val : !llvm.ptr<3>, vector<2xi16>
  llvm.return
}

// -----

llvm.func @triton_gen.sub_group_block_write(%ptr: !llvm.ptr<1>, %val : i32) {
  // CHECK:     llvm.func @triton_gen.sub_group_block_write(%arg0: !llvm.ptr<1>, %arg1: i32) {
  // CHECK:       llvm.call spir_funccc @_Z30intel_sub_group_block_write_uiPU3AS1jj(%arg0, %arg1) {{.*}} : (!llvm.ptr<1>, i32) -> ()
  triton_gen.sub_group_block_write %ptr, %val : !llvm.ptr<1>, i32
  llvm.return
}

// -----

llvm.func @triton_gen.predicated_load(%ptr : !llvm.ptr<1>, %alignment : i64, %predicate : i1, %default_value : i32) {
  // CHECK:     llvm.func @triton_gen.predicated_load(%arg0: !llvm.ptr<1>, %arg1: i64, %arg2: i1, %arg3: i32) {
  // CHECK:       %0 = llvm.call spir_funccc @llvm.genx.GenISA.PredicatedLoad.i32.p1i32.i32(%arg0, %arg1, %arg2, %arg3) {{.*}} : (!llvm.ptr<1>, i64, i1, i32) -> i32
  %0 = triton_gen.predicated_load %ptr, %alignment, %predicate, %default_value : !llvm.ptr<1>, i64, i1, i32 -> i32
  llvm.return
}
