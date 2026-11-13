// RUN: triton-opt -convert-tritongen-to-spirv -split-input-file %s | FileCheck %s

llvm.func @triton_gen.barrier.none() {
  // CHECK-LABEL: @triton_gen.barrier.none
  // CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <None>
  triton_gen.barrier { mem_fence = None }
  llvm.return
}

// -----

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

llvm.func @triton_gen.barrier.local_and_global() {
  // CHECK-LABEL: @triton_gen.barrier.local_and_global
  // CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory|CrossWorkgroupMemory>
  triton_gen.barrier { mem_fence = LocalAndGlobal }
  llvm.return
}
