
// total 16x32x1024
//              it has 1 workgroup. each workgroup caculates a [16x32 = 16x1024 * 1024x32] block
// each work-group has 4  subgroup. each  subgroup caculates a [ 8x16 = 8x1024 * 1024x16] block
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
tt.func @test_kernel(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c8 = arith.constant 8 : i32
    %c1024 = arith.constant 1024 : i32
    %m = arith.constant 16 : i64
    %n = arith.constant 32 : i64
    %k = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %id = gpu.subgroup_id  : index
    %sgid = arith.index_cast %id : index to i32
    %c2 = arith.constant 2: i32


    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
      %ax = arith.constant 0 : i32
      %ay = arith.divsi %sgid, %c2 : i32
      %offsetax = arith.muli %ax, %c16 : i32
      %offsetay = arith.muli %ay, %c8 : i32
    %aPtr = tt.make_tensor_ptr %arg0, [%m, %k], [%k, %c1_i64], [%offsetay, %offsetax] {order = array<i32: 1, 0>} : <tensor<8x16xf16>, 1>
      %bx = arith.remsi %sgid, %c2 : i32
      %by = arith.constant 0 : i32
      %offsetbx = arith.muli %bx, %c16 : i32
      %offsetby = arith.muli %by, %c16 : i32
    %bPtr = tt.make_tensor_ptr %arg1, [%k, %n], [%n, %c1_i64], [%offsetby, %offsetbx] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %6:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cst, %subA = %aPtr, %subB = %bPtr) -> (tensor<8x16xf32>, !tt.ptr<tensor<8x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>) : i32 {
      %a = tt.load %subA {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x16xf16>, 1> -> tensor<8x16xf16>
      %b = tt.load %subB {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, DotB = true} : !tt.ptr<tensor<16x16xf16>, 1> -> tensor<16x16xf16>
      %c = tt.dot %a, %b, %arg4 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %30 = tt.advance %subA, [%c0, %c16] : <tensor<8x16xf16>, 1>
      %31 = tt.advance %subB, [%c16, %c0] : <tensor<16x16xf16>, 1>
      scf.yield %c, %30, %31 : tensor<8x16xf32>, !tt.ptr<tensor<8x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>
    }
    %value = arith.truncf %6#0 : tensor<8x16xf32> to tensor<8x16xf16>
      %cx = arith.remsi %sgid, %c2 : i32
      %cy = arith.divsi %sgid, %c2 : i32
      %offsetcx = arith.muli %cx, %c16 : i32
      %offsetcy = arith.muli %cy, %c8 : i32
    %cPtr = tt.make_tensor_ptr %arg2, [%m, %n], [%n, %c1_i64], [%offsetcy, %offsetcx] {order = array<i32: 1, 0>} : <tensor<8x16xf16>, 1>
    tt.store %cPtr, %value {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<8x16xf16>, 1>, tensor<8x16xf16>
    tt.return

}
}
