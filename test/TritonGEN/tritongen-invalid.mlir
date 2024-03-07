// RUN: triton-opt -split-input-file -verify-diagnostics %s

llvm.func @triton_gen.fptofp(%a : i32) {
  // expected-error @+1 {{custom op 'triton_gen.conv.fptofp' invalid kind of type specified}}
  %0 = triton_gen.conv.fptofp %a {roundingMode = rte} : i32 to i16
  llvm.return
}

// -----

llvm.func @triton_gen.fptofp(%a : f32) {
  // expected-error @+1 {{'triton_gen.conv.fptofp' op expecting first argument and result size to be different}}
  %0 = triton_gen.conv.fptofp %a {roundingMode = rte} : f32 to f32
  llvm.return
}

// -----

llvm.func @triton_gen.fptofp(%a : f32) {
  // expected-error @+1 {{'triton_gen.conv.fptofp' op expecting rounding mode for truncation}}
  %0 = triton_gen.conv.fptofp %a : f32 to f16
  llvm.return
}
