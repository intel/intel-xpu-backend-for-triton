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

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 32>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // expected-error @+2 {{'triton_gen.sub_group_reduce' op expecting floating point type for floating point reduction}}
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = triton_gen.sub_group_reduce fsum %0 {size = 16} : i32
    llvm.return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 32>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // expected-error @+2 {{'triton_gen.sub_group_reduce' op expecting integer type for integer reduction}}
    %0 = llvm.mlir.constant(0.0 : f32) : f32
    %1 = triton_gen.sub_group_reduce sum %0 {size = 16} : f32
    llvm.return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // expected-error @+2 {{'triton_gen.sub_group_reduce' op expecting size to be a power of 2 between 1 and subgroup size}}
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = triton_gen.sub_group_reduce sum %0 {size = 0} : i32
    llvm.return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // expected-error @+2 {{'triton_gen.sub_group_reduce' op expecting size to be a power of 2 between 1 and subgroup size}}
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = triton_gen.sub_group_reduce sum %0 {size = 32} : i32
    llvm.return
  }
}

// -----

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  llvm.func @triton_gen.sub_group_reduce() {
    // expected-error @+2 {{'triton_gen.sub_group_reduce' op expecting size to be a power of 2 between 1 and subgroup size}}
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = triton_gen.sub_group_reduce sum %0 {size = 6} : i32
    llvm.return
  }
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
  // expected-error @+1 {{'triton_gen.dpas' op the element type for 1st operand (C) and the result should be f32}}
  %0 = triton_gen.dpas %c, %a, %b {pa=f16, pb=f16, rc=8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
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
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op transpose and vnni transform are mutually exclusive}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=true, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockload' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op tile_width for 32 bit elements should be equal to systolic depth, i.e., 8 elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=5, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<5xf32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op tile_width for 16 bit elements should be equal to systolic depth times 2, i.e., 16 elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xf16>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op tile_width for 8 bit elements should be equal to systolic depth times 4, i.e., 32 elements}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi8>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockload' op expecting tile_height to be 1, 2, 4, 8, 16, or 32}}
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=64, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xf32>
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=64, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op transpose and vnni transform are mutually exclusive}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=true, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockstore' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xf32>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 32 bit elements should be equal to systolic depth, i.e., 8 elements}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xf32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xf16>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 16 bit elements should be equal to systolic depth times 2, i.e., 16 elements}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xf16>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op tile_width for 8 bit elements should be equal to systolic depth times 4, i.e., 32 elements}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xf32>) {
  // expected-error @+1 {{'triton_gen.2Dblockstore' op expecting tile_height to be 1, 2, 4, 8, 16, or 32}}
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=8, tile_height=64, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op transpose and vnni transform are mutually exclusive}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=4, tile_height=1, v_blocks=1, transpose=true, vnni_transform=true, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op tile_width for 32 bit elements should be equal to systolic depth, i.e., 8 elements}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=5, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op tile_width for 16 bit elements should be equal to systolic depth times 2, i.e., 16 elements}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op tile_width for 8 bit elements should be equal to systolic depth times 4, i.e., 32 elements}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=4, tile_height=1, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

// -----

llvm.func @matrix_2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'triton_gen.2Dblockprefetch' op expecting tile_height to be 1, 2, 4, 8, 16, or 32}}
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=64, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}
