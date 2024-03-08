// RUN: triton-opt -split-input-file -verify-diagnostics %s

llvm.func @triton_gen.fptofp(%a : i32) {
  // expected-error @+1 {{custom op 'triton_gen.fptofp' invalid kind of type specified}}
  %0 = triton_gen.fptofp %a {roundingMode = rte} : i32 to i16
  llvm.return
}

// -----

llvm.func @triton_gen.fptofp(%a : f32) {
  // expected-error @+1 {{'triton_gen.fptofp' op expecting first argument and result size to be different}}
  %0 = triton_gen.fptofp %a {roundingMode = rte} : f32 to f32
  llvm.return
}

// -----

llvm.func @triton_gen.fptofp(%a : f32) {
  // expected-error @+1 {{'triton_gen.fptofp' op expecting rounding mode for truncation}}
  %0 = triton_gen.fptofp %a : f32 to f16
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting repeat count to be 1, 2, 4, or 8}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=6} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting precision of matrix A and B to be the same}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=u8, rc=8} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi8>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op 1st operand (C) and result (D) should have the same type}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=8} : (vector<8xi8>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<16xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op the dimension for 1st operand (C) and result (D) should match repeat count}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=8} : (vector<16xi32>, vector<16xi8>, vector<32xi8>) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi8>, %b : vector<8xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op 2nd operand (A) bit-size should be repeat count times 16}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=8} : (vector<8xi32>, vector<8xi8>, vector<8xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<16xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op 3rd operand (B) bit-size should be systolic depth (8) times 32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=8} : (vector<8xi32>, vector<16xi8>, vector<16xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi8>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be i32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s8, pb=s8, rc=8} : (vector<8xi8>, vector<16xi8>, vector<32xi8>) -> vector<8xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<8xf16>, %b : vector<16xf16>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting precision type to be tf32, bf16, fp16, u8, or s8}}
  %0 = triton_gen.dpas %c, %a, %b {pa=s4, pb=s4, rc=8} : (vector<8xf32>, vector<8xf16>, vector<16xf16>) -> vector<8xf32>
  llvm.return
}
