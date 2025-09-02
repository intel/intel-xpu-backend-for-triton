// RUN: triton-opt -convert-tritongen-to-spirv -split-input-file %s | FileCheck %s

llvm.func @triton_gen.barrier.local() {
  // CHECK-LABEL: @triton_gen.barrier.local
  // CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
  triton_gen.barrier { mem_fence = Local }
  llvm.return
}

// -----

llvm.func @triton_gen.barrier.global() {
  // CHECK-LABEL: @triton_gen.barrier.global
  // CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|CrossWorkgroupMemory>
  triton_gen.barrier { mem_fence = Global }
  llvm.return
}

// -----

llvm.func @triton_gen.split_barrier_arrive() {
  // CHECK-LABEL: triton_gen.split_barrier_arrive() {
  // CHECK:       spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <None>
  triton_gen.split_barrier_arrive {execution_scope=WorkGroup, memory_scope=WorkGroup}
  llvm.return
}

// -----

llvm.func @triton_gen.split_barrier_wait() {
  // CHECK-LABEL: triton_gen.split_barrier_wait() {
  // CHECK:       spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <None>
  triton_gen.split_barrier_wait {execution_scope=WorkGroup, memory_scope=WorkGroup}
  llvm.return
}
