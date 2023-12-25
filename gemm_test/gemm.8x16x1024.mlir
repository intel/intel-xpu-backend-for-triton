// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_8x1024xf16 : memref<8x1024xf16> = dense<0.0>
  memref.global "private" @__constant_1024x16xf16_ : memref<1024x16xf16> = dense<0.0>
  memref.global "private" @__constant_8x16xf16 : memref<8x16xf16> = dense<0.0>
  func.func @test(%arg0: memref<8x1024xf16>, %arg1: memref<1024x16xf16>) -> memref<8x16xf16> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x1024xf16>
    memref.copy %arg0, %memref : memref<8x1024xf16> to memref<8x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x16xf16>
    memref.copy %arg1, %memref_0 : memref<1024x16xf16> to memref<1024x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x1024xf16>, %memref_0 : memref<1024x16xf16>, %memref_1 : memref<8x16xf16>)
    gpu.dealloc  %memref : memref<8x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x16xf16>
    return %memref_1 : memref<8x16xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x1024xf16>, %arg1: memref<1024x16xf16>, %arg2: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index
      //%0 = gpu.block_id  x
      //%1 = gpu.block_id  y
      //%2 = arith.muli %0, %c8 : index
      //%3 = arith.muli %1, %c16 : index
      %cst = arith.constant dense<0.0> : vector<128xf32>
      %cast = vector.shape_cast %cst : vector<128xf32> to vector<8x16xf32>
      // each work-group has 1 subgroup. the subgroup caculates a [8x16 = 8x1024 * 1024x16] block
        %7 = xegpu.create_nd_tdesc %arg0[%c0, %c0] {mode=vc}: memref<8x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
        %8 = xegpu.create_nd_tdesc %arg1[%c0, %c0]  {mode=vc}: memref<1024x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %6:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cast, %subA = %7, %subB = %8) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %9 = xegpu.load_nd %subA  {mode=vc, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %10 = xegpu.load_nd %subB  {mode=vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %11 = xegpu.dpas %9, %10, %arg4 {mode=vc}: vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %12 = xegpu.update_nd_offset %subA, [%c0, %c16] {mode=vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %13 = xegpu.update_nd_offset %subB, [%c16, %c0] {mode=vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        scf.yield %11, %12, %13: vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }
      %new = arith.truncf %6#0 : vector<8x16xf32> to vector<8x16xf16>
      %4 = xegpu.create_nd_tdesc %arg2[%c0, %c0] {mode = vc} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %new, %4 {mode = vc}: vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.get_global @__constant_8x1024xf16 : memref<8x1024xf16>
    %1 = memref.get_global @__constant_1024x16xf16_ : memref<1024x16xf16>
    %ref = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    %init = arith.constant 0.0 : f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    // A matrix: row-major, start from 0.0, increase 0.01 per element
    // B matrix: A matrix + 1.0
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %int0 = arith.index_cast %arg0 : index to i16
        %int1 = arith.index_cast %arg1 : index to i16
        %c128_i16 = arith.constant 128 : i16
        %idx0 = arith.muli %int0, %c128_i16 : i16
        %idx1 = arith.addi %int1, %idx0 : i16
        %fp = arith.uitofp %idx1 : i16 to f16
        %cst100 = arith.constant 1000.0 : f16
        %val0 = arith.divf %fp, %cst100 : f16
        memref.store %val0, %0[%arg0, %arg1] : memref<8x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %int0 = arith.index_cast %arg0 : index to i16
        %int1 = arith.index_cast %arg1 : index to i16
        %idx1 = arith.addi %int1, %int0 : i16
        %fp = arith.uitofp %idx1 : i16 to f16
        %cst100 = arith.constant 100.0 : f16
        %val0 = arith.divf %fp, %cst100 : f16
        memref.store %val0, %1[%arg0, %arg1] : memref<1024x16xf16>
      }
    }
    // caculate the result C matrix
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        //%acc = memref.load %ref[%arg0, %arg1] : memref<8x16xf16>
        %acc = arith.constant 0.0 : f32
        %res = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %acc) -> f32 {
          %a = memref.load %0[%arg0, %arg2] : memref<8x1024xf16>
          %b = memref.load %1[%arg2, %arg1] : memref<1024x16xf16>
          %c = arith.mulf %a, %b : f16
          %cc = arith.extf %c : f16 to f32
          %ccc = arith.addf %cc, %arg3 : f32
          scf.yield %ccc : f32
        }
        %new = arith.truncf %res : f32 to f16
        memref.store %new, %ref[%arg0, %arg1] : memref<8x16xf16>
      }
    }

    %2 = call @test(%0, %1) : (memref<8x1024xf16>, memref<1024x16xf16>) -> memref<8x16xf16>
    %cast = memref.cast %2 : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_ref = memref.cast %ref : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF16(%cast_ref) : (memref<*xf16>) -> ()
    // CHECK:   [ALLCLOSE: TRUE]
    call @printAllcloseF16(%cast, %cast_ref) : (memref<*xf16>, memref<*xf16>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
}
