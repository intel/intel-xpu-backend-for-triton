// RUN: triton-opt --canonicalize --cse %s | FileCheck %s

// COM: The write has no result, so only its MemWrite effect stops canonicalize/CSE erasing it.
// CHECK-LABEL: llvm.func @f
// CHECK: triton_gen.sub_group_block_write
llvm.func @f(%ptr: !llvm.ptr<1>, %v: vector<2xi16>) {
  triton_gen.sub_group_block_write %ptr, %v : !llvm.ptr<1>, vector<2xi16>
  llvm.return
}
