// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// Test basic 32-bit atomic CAS with shared memory allocation
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_i32_global
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> i32
  // CHECK-SAME: attributes {intel_reqd_sub_group_size = 32 : i32, reqd_work_group_size = array<i32: 32, 1, 1>}
  tt.func @test_atomic_cas_i32_global(%ptr: !tt.ptr<i32, 1>, %cmp: i32, %val: i32) -> i32 {

    // CHECK: %[[TID_CALL:.*]] = llvm.call spir_funccc @_Z12get_local_idj
    // CHECK: %[[TID:.*]] = llvm.trunc %[[TID_CALL]] : i64 to i32
    // CHECK: %[[THREAD_MASK:.*]] = llvm.and %[[TID]], %{{.*}} : i32
    // CHECK: %[[IS_THREAD_0:.*]] = llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.cond_br %{{.*}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}

    // CHECK: %[[C21:.*]] = llvm.bitcast %arg1 : i32 to i32
    // CHECK: %[[C22:.*]] = llvm.bitcast %arg2 : i32 to i32
    // CHECK: %[[RESULT:.*]] = llvm.cmpxchg %arg0, %[[C21]], %[[C22]] acq_rel monotonic : !llvm.ptr<1>, i32
    // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[RESULT]][0] : !llvm.struct<(i32, i1)>

    // CHECK: %[[SMEM_ADDR:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK: %[[SMEM_PTR:.*]] = llvm.getelementptr %[[SMEM_ADDR]]
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<3>

    // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {convergent, no_unwind, will_return} : (i32) -> ()

    // CHECK: %[[FINAL_RESULT:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
    // CHECK: llvm.return %[[FINAL_RESULT]] : i32

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<i32, 1>, i32, i32) -> i32
    tt.return %0 : i32
  }
}

// -----

// Test 64-bit atomic CAS
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_i64_global
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: i64, %arg2: i64, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> i64
  // CHECK-SAME: attributes {intel_reqd_sub_group_size = 32 : i32, reqd_work_group_size = array<i32: 32, 1, 1>}
  tt.func @test_atomic_cas_i64_global(%ptr: !tt.ptr<i64, 1>, %cmp: i64, %val: i64) -> i64 {

    // CHECK: %[[TID_CALL:.*]] = llvm.call spir_funccc @_Z12get_local_idj
    // CHECK: %[[TID:.*]] = llvm.trunc %[[TID_CALL]] : i64 to i32
    // CHECK: %[[THREAD_MASK:.*]] = llvm.and %[[TID]], %{{.*}} : i32
    // CHECK: %[[IS_THREAD_0:.*]] = llvm.icmp "eq" %{{.*}}, %{{.*}} : i32
    // CHECK: llvm.cond_br %{{.*}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}

    // CHECK: %[[C21:.*]] = llvm.bitcast %arg1 : i64 to i64
    // CHECK: %[[C22:.*]] = llvm.bitcast %arg2 : i64 to i64
    // CHECK: %[[RESULT:.*]] = llvm.cmpxchg %arg0, %[[C21]], %[[C22]] acq_rel monotonic : !llvm.ptr<1>, i64
    // CHECK: %[[EXTRACT:.*]] = llvm.extractvalue %[[RESULT]][0] : !llvm.struct<(i64, i1)>

    // CHECK: %[[SMEM_ADDR:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK: %[[SMEM_PTR:.*]] = llvm.getelementptr %[[SMEM_ADDR]]
    // CHECK: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<3>

    // CHECK: llvm.call spir_funccc @_Z7barrierj(%{{.*}}) {convergent, no_unwind, will_return} : (i32) -> ()

    // CHECK: %[[FINAL_RESULT:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i64
    // CHECK: llvm.return %[[FINAL_RESULT]] : i64

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<i64, 1>, i64, i64) -> i64
    tt.return %0 : i64
  }
}

// -----
// Test tensor atomic CAS
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_tensor
  // CHECK-SAME: (%arg0: !llvm.struct<(ptr<1>)>, %arg1: !llvm.struct<(i32)>, %arg2: !llvm.struct<(i32)>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> !llvm.struct<(i32)>
  // CHECK-SAME: attributes {intel_reqd_sub_group_size = 32 : i32, reqd_work_group_size = array<i32: 32, 1, 1>}
  tt.func @test_atomic_cas_tensor(%ptr: tensor<32x!tt.ptr<i32, 1>, #blocked>,
                                  %cmp: tensor<32xi32, #blocked>,
                                  %val: tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {

    // CHECK: %[[PTR:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>)>
    // CHECK: %[[CMP:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK: %[[VAL:.*]] = llvm.extractvalue %arg2[0] : !llvm.struct<(i32)>

    // CHECK: %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TID_CALL:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[C0_2]])
    // CHECK-SAME: {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : (i32) -> i64
    // CHECK: %[[TID:.*]] = llvm.trunc %[[TID_CALL]] : i64 to i32

    // CHECK: %[[C31:.*]] = llvm.mlir.constant(31 : i32) : i32
    // CHECK: %[[TID_MASKED:.*]] = llvm.and %[[TID]], %[[C31]] : i32

    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : i32) : i32

    // CHECK: %[[CMP_CAST:.*]] = llvm.bitcast %[[CMP]] : i32 to i32
    // CHECK: %[[VAL_CAST:.*]] = llvm.bitcast %[[VAL]] : i32 to i32

    // CHECK: %[[ATOMIC:.*]] = llvm.cmpxchg %[[PTR]], %[[CMP_CAST]], %[[VAL_CAST]] acq_rel monotonic : !llvm.ptr<1>, i32
    // CHECK: %[[RESULT:.*]] = llvm.extractvalue %[[ATOMIC]][0] : !llvm.struct<(i32, i1)>

    // CHECK: %[[RESULT_CAST:.*]] = llvm.bitcast %[[RESULT]] : i32 to i32
    // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
    // CHECK: %[[PACKED:.*]] = llvm.insertvalue %[[RESULT_CAST]], %[[UNDEF]][0] : !llvm.struct<(i32)>
    // CHECK: llvm.return %[[PACKED]] : !llvm.struct<(i32)>

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (tensor<32x!tt.ptr<i32, 1>, #blocked>, tensor<32xi32, #blocked>, tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked>
    tt.return %0 : tensor<32xi32, #blocked>
  }
}

// -----

// Test 16-bit atomic CAS with hardware support
module attributes {ttig.support_16bit_atomics = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_i16_hw_support
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: i16, %arg2: i16, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> i16
  // CHECK-SAME: attributes {intel_reqd_sub_group_size = 32 : i32, reqd_work_group_size = array<i32: 32, 1, 1>}
  tt.func @test_atomic_cas_i16_hw_support(%ptr: !tt.ptr<i16, 1>, %cmp: i16, %val: i16) -> i16 {

    // CHECK: %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[C0_2:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[TID_CALL:.*]] = llvm.call spir_funccc @_Z12get_local_idj(%[[C0_2]])
    // CHECK-SAME: {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return} : (i32) -> i64
    // CHECK: %[[TID:.*]] = llvm.trunc %[[TID_CALL]] : i64 to i32
    // CHECK: %[[C31:.*]] = llvm.mlir.constant(31 : i32) : i32
    // CHECK: %[[THREAD_MASK:.*]] = llvm.and %[[TID]], %[[C31]] : i32

    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[C0_3:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[C0_4:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[NEG1_1:.*]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[AND1:.*]] = llvm.and %[[THREAD_MASK]], %[[NEG1_1]] : i32
    // CHECK: %[[CMP1:.*]] = llvm.icmp "eq" %[[AND1]], %[[C0_1]] : i32
    // CHECK: %[[NEG1_2:.*]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[AND2:.*]] = llvm.and %[[C0_3]], %[[NEG1_2]] : i32
    // CHECK: %[[CMP2:.*]] = llvm.icmp "eq" %[[AND2]], %[[C0_1]] : i32
    // CHECK: %[[AND_CMP12:.*]] = llvm.and %[[CMP1]], %[[CMP2]] : i1
    // CHECK: %[[NEG1_3:.*]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[AND3:.*]] = llvm.and %[[C0_4]], %[[NEG1_3]] : i32
    // CHECK: %[[CMP3:.*]] = llvm.icmp "eq" %[[AND3]], %[[C0_1]] : i32
    // CHECK: %[[FINAL_CMP:.*]] = llvm.and %[[AND_CMP12]], %[[CMP3]] : i1

    // CHECK: %[[C0_I16:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: llvm.cond_br %[[FINAL_CMP]], ^bb1, ^bb2(%[[C0_I16]] : i16)

    // CHECK: ^bb1:
    // CHECK: %[[CMP_CAST:.*]] = llvm.bitcast %arg1 : i16 to i16
    // CHECK: %[[VAL_CAST:.*]] = llvm.bitcast %arg2 : i16 to i16

    // Direct atomic compare-exchange on 16-bit value (hardware supported)
    // CHECK: %[[CMPXCHG:.*]] = llvm.cmpxchg %arg0, %[[CMP_CAST]], %[[VAL_CAST]] acq_rel monotonic : !llvm.ptr<1>, i16
    // CHECK: %[[RESULT:.*]] = llvm.extractvalue %[[CMPXCHG]][0] : !llvm.struct<(i16, i1)>
    // CHECK: llvm.br ^bb2(%[[RESULT]] : i16)

    // CHECK: ^bb2(%[[PHI_RESULT:.*]]: i16):
    // CHECK: %[[RESULT_CAST:.*]] = llvm.bitcast %[[PHI_RESULT]] : i16 to i16
    // CHECK: %[[C0_5:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[SMEM_ADDR:.*]] = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[SMEM_ADDR]][%[[C0_5]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8
    // CHECK: %[[SMEM_PTR:.*]] = llvm.bitcast %[[GEP]] : !llvm.ptr<3> to !llvm.ptr<3>
    // CHECK: llvm.cond_br %[[FINAL_CMP]], ^bb3, ^bb4

    // CHECK: ^bb3:
    // CHECK: llvm.store %[[RESULT_CAST]], %[[SMEM_PTR]] : i16, !llvm.ptr<3>
    // CHECK: llvm.br ^bb4

    // CHECK: ^bb4:
    // CHECK: %[[BARRIER_FLAG:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: llvm.call spir_funccc @_Z7barrierj(%[[BARRIER_FLAG]]) {convergent, no_unwind, will_return} : (i32) -> ()
    // CHECK: %[[FINAL_RESULT:.*]] = llvm.load %[[SMEM_PTR]] : !llvm.ptr<3> -> i16
    // CHECK: llvm.return %[[FINAL_RESULT]] : i16

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<i16, 1>, i16, i16) -> i16
    tt.return %0 : i16
  }
}

// -----

// Test 16-bit atomic CAS with emulation (no hardware support)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_i16_emulated
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: i16, %arg2: i16, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> i16
  tt.func @test_atomic_cas_i16_emulated(%ptr: !tt.ptr<i16, 1>, %cmp: i16, %val: i16) -> i16 {

    // Same emulation pattern as hardware support version but without hardware optimization
    // CHECK: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi16>
    // CHECK: llvm.icmp "eq" %{{.*}}, %{{.*}} : i16
    // CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.ptr<1>, i32

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<i16, 1>, i16, i16) -> i16
    tt.return %0 : i16
  }
}

// -----

// Test f16 atomic CAS with emulation
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_f16_emulated
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: f16, %arg2: f16, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> f16
  tt.func @test_atomic_cas_f16_emulated(%ptr: !tt.ptr<f16, 1>, %cmp: f16, %val: f16) -> f16 {

    // CHECK: %[[I16_ZERO:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: llvm.cond_br %{{.*}}, ^bb1, ^bb4(%[[I16_ZERO]] : i16)

    // CHECK: %[[CMP_CAST:.*]] = llvm.bitcast %arg1 : f16 to i16
    // CHECK: %[[VAL_CAST:.*]] = llvm.bitcast %arg2 : f16 to i16

    // CHECK: %{{.*}} = llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[CURRENT_I16:.*]] = llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi16>

    // CHECK: %[[I16_EQ:.*]] = llvm.icmp "eq" %[[CURRENT_I16]], %[[CMP_CAST]] : i16
    // CHECK: llvm.cond_br %[[I16_EQ]], ^bb3, ^bb4(%[[CURRENT_I16]] : i16)

    // CHECK: %[[NEW_I16:.*]] = llvm.insertelement %[[VAL_CAST]], %{{.*}}[%{{.*}} : i32] : vector<2xi16>
    // CHECK: %{{.*}} = llvm.bitcast %[[NEW_I16]] : vector<2xi16> to i32
    // CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.ptr<1>, i32

    // CHECK: %[[RESULT_F16:.*]] = llvm.bitcast %{{.*}} : i16 to f16
    // CHECK: llvm.store %[[RESULT_F16]], %{{.*}} : f16, !llvm.ptr<3>
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> f16

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<f16, 1>, f16, f16) -> f16
    tt.return %0 : f16
  }
}

// -----

// Test bf16 atomic CAS with emulation
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 2 : i32} {

  // CHECK-DAG: llvm.func spir_funccc @_Z7barrierj(i32) attributes {convergent, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func spir_kernelcc @test_atomic_cas_bf16_emulated
  // CHECK-SAME: (%arg0: !llvm.ptr<1>, %arg1: bf16, %arg2: bf16, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) -> bf16
  tt.func @test_atomic_cas_bf16_emulated(%ptr: !tt.ptr<bf16, 1>, %cmp: bf16, %val: bf16) -> bf16 {

    // CHECK: %[[I16_ZERO:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: llvm.cond_br %{{.*}}, ^bb1, ^bb4(%[[I16_ZERO]] : i16)

    // CHECK: %[[CMP_CAST:.*]] = llvm.bitcast %arg1 : bf16 to i16
    // CHECK: %[[VAL_CAST:.*]] = llvm.bitcast %arg2 : bf16 to i16

    // CHECK: %{{.*}} = llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[CURRENT_I16:.*]] = llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi16>

    // CHECK: %[[I16_EQ:.*]] = llvm.icmp "eq" %[[CURRENT_I16]], %[[CMP_CAST]] : i16
    // CHECK: llvm.cond_br %[[I16_EQ]], ^bb3, ^bb4(%[[CURRENT_I16]] : i16)

    // CHECK: %[[NEW_I16:.*]] = llvm.insertelement %[[VAL_CAST]], %{{.*}}[%{{.*}} : i32] : vector<2xi16>
    // CHECK: %{{.*}} = llvm.bitcast %[[NEW_I16]] : vector<2xi16> to i32
    // CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.ptr<1>, i32

    // CHECK: %[[RESULT_BF16:.*]] = llvm.bitcast %{{.*}} : i16 to bf16
    // CHECK: llvm.store %[[RESULT_BF16]], %{{.*}} : bf16, !llvm.ptr<3>
    // CHECK: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> bf16

    %0 = tt.atomic_cas acq_rel, cta, %ptr, %cmp, %val : (!tt.ptr<bf16, 1>, bf16, bf16) -> bf16
    tt.return %0 : bf16
  }
}
