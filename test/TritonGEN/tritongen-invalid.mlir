// RUN: triton-opt -split-input-file -verify-diagnostics %s

llvm.func @triton_gen.duplicated_cache_controls(%arg0: !llvm.ptr) {
  // expected-error @+1 {{'triton_gen.decoration_cache_controls' cannot specify more than one cache control decoration of the same nature for the same cache level}}
  %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 0>>} : !llvm.ptr -> i32
  llvm.return
}

// -----

llvm.func @triton_gen.illegal_cache_controls_attr(%arg0: !llvm.ptr) {
  // expected-error @+1 {{'triton_gen.decoration_cache_controls' only accepts LoadCacheControlDecorationAttr and StoreCacheControlDecorationAttr attributes}}
  %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<1 : i32>} : !llvm.ptr -> i32
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting repeat count to be 1, 2, 4, or 8}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=16} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting precision of matrix A and B to be the same}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=f16, rc=8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi8>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op 1st operand (C) and result (D) should have the same type}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi8>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<16xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the dimension for 1st operand (C) and result (D) should match repeat count}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<16xi32>, vector<8xi16>, vector<8xi32>) -> vector<16xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<16xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the dimension for the 3rd operand (B) should match the systolic depth of 8}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<8xi16>, vector<16xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi8>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be i32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi8>, vector<8xi16>, vector<8xi32>) -> vector<8xi8>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be f16 or f32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=f16, pb=f16, rc=8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf16>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be bf16 or f32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=bf16, pb=bf16, rc=8} : (vector<8xf16>, vector<8xi16>, vector<8xi32>) -> vector<8xf16>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf16>, %a : vector<4xf32>, %b : vector<8xf32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be f32}}
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf16>, vector<4xf32>, vector<8xf32>) -> vector<8xf16>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<8xf32>, %b : vector<8xf32>) {
  // expected-error @+1 {{'triton_gen.dpas' op the dimension for the 2nd operand (A) should be equal to half of the repeat count}}
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<4xi16>, %b : vector<8xf32>) {
  // expected-error @+1 {{'triton_gen.dpas' op 2nd operand (A) element type should be f32 or i32 when the precision type is tf32}}
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<4xi16>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// -----


llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<4xf32>, %b : vector<8xi16>) {
  // expected-error @+1 {{'triton_gen.dpas' op 3rd operand (B) element type should be f32 or i32 when the precision type is tf32}}
  %0 = triton_gen.dpas %c, %a, %b {pa = tf32, pb = tf32, rc = 8} : (vector<8xf32>, vector<4xf32>, vector<8xi16>) -> vector<8xf32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<4xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op 2nd operand (A) should have the same number of elements as repeat count}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<4xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi8>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op 2nd operand (A) element type should be i16 when the precision type is not tf32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<8xi8>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xf32>) {
  // expected-error @+1 {{'triton_gen.dpas' op 3rd operand (B) element type should be i32 when the precision type is not tf32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=f16, pb=f16, rc=8} : (vector<8xf32>, vector<8xi16>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

// -----

llvm.func @triton_gen.dpas(%c : vector<8xf32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // expected-error @+1 {{'triton_gen.dpas' op expecting precision type to be tf32, bf16, fp16, u8, or s8}}
  %0 = triton_gen.dpas %c, %a, %b {pa=i4, pb=i4, rc=8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op result size of 256 bits does not match the expected size of 128 bits}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op transpose and vnni_transform are mutually exclusive}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=16, tile_height=8, v_blocks=1, transpose=true, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<2xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting tile_width to be between 4 and 64}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=1, tile_height=32, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<1xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting elem_size_in_bits * tile_width * v_blocks <= 512}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=4, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op transpose is only supported for 32 bit elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=true, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op vnni_transform is only supported for 8 and 16 bit elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 2nd operand (base width) should be <= 24 bits}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 2nd operand (base width) should be >= 64}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(65 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 2nd operand (base width) should be aligned to MAX(4, element_size)}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_height = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 3rd operand (base height) should be <= 24 bits}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32) {
  %base_pitch = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 4th operand (base pitch) should be <= 24 bits}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32) {
  %base_pitch = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 4th operand (base pitch) should be >= 64}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32) {
  %base_pitch = llvm.mlir.constant(65 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 4th operand (base pitch) should be a multiple of 16 bytes}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(68 : i32) : i32
  %base_pitch = llvm.mlir.constant(64 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %y : i32) {
  %x = llvm.mlir.constant(1 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 5th operand (x) should be a multiple of 4 for 8 bit elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %y : i32) {
  %x = llvm.mlir.constant(1 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 5th operand (x) should be a multiple of 2 for 16 bit elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64, tile_width=4, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting tile shape to be power of two}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=48, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<12xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting tile_width to be between 1-64}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=128, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<32xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting tile_height to be between 1-32}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=64, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<64xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op tile_width when vnni_transform is true should be equal to subgroup size (16 elements)}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<1xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting result element type to be 32 bits}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xi16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<64xi8>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting tile_height to be between 1 and 8}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<64xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting tile_width to be between 4 and 64}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=2, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting v_blocks to be 1}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=2, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}


// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_width = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 2nd operand (base width) should be <= 24 bits}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_width = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 2nd operand (base width) should be >= 64}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_height = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 3rd operand (base height) should be <= 24 bits}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_pitch = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 4th operand (base pitch) should be <= 24 bits}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_pitch = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 4th operand (base pitch) should be >= 64}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}
// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<8xi8>) {
  %base_width = llvm.mlir.constant(68 : i32) : i32
  %base_pitch = llvm.mlir.constant(64 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<32xi16>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=64, tile_width=4, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<32xi16>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 8 bit elements should be equal to 16 or 32}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=8, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 16 bit elements should be equal to 16}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 32 bit elements should be equal to 16}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 2nd operand (base width) should be <= 24 bits}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 2nd operand (base width) should be >= 64}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_pitch : i32, %x : i32, %y : i32) {
  %base_height = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 3rd operand (base height) should be <= 24 bits}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32) {
  %base_pitch = llvm.mlir.constant(16777217 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 4th operand (base pitch) should be <= 24 bits}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %x : i32, %y : i32) {
  %base_pitch = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 4th operand (base pitch) should be >= 64}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(68 : i32) : i32
  %base_pitch = llvm.mlir.constant(64 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64, tile_width=4, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting tile_height to be between 1-32}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=64, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting v_blocks to be between 1-4}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=8, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting elem_size_in_bits * tile_width * v_blocks <= 512}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting tile_width to be between 4 and 64}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=1, tile_height=32, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}
