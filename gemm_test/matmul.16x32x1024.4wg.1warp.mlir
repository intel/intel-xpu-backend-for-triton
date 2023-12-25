
// total 16x32x1024
//              it has 4 workgroup. each workgroup caculates a [ 8x16 = 8x1024 * 1024x16] block
// each work-group has 1  subgroup. each  subgroup caculates a [ 8x16 = 8x1024 * 1024x16] block
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
tt.func @test_kernel(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c8 = arith.constant 8 : i32
    %c1024 = arith.constant 1024 : i32
    %m = arith.constant 16 : i64
    %n = arith.constant 32 : i64
    %k = arith.constant 1024 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
    %pid =  tt.get_program_id x : i32
    %c2 = arith.constant 2: i32
    %pidm = arith.divsi %pid, %c2 : i32
    %pidn = arith.remsi %pid, %c2 : i32
    %offsetX = arith.muli %pidn, %c16 : i32
    %offsetYA = arith.muli %pidm, %c8 : i32
    %offsetYB = arith.muli %pidm, %c16 : i32
    %aPtr = tt.make_tensor_ptr %arg0, [%m, %k], [%k, %c1_i64], [%offsetYA, %offsetX] {order = array<i32: 1, 0>} : <tensor<8x16xf16>, 1>
    %bPtr = tt.make_tensor_ptr %arg1, [%k, %n], [%n, %c1_i64], [%offsetYB, %offsetX] {order = array<i32: 1, 0>} : <tensor<16x16xf16>, 1>
    %6:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cst, %subA = %aPtr, %subB = %bPtr) -> (tensor<8x16xf32>, !tt.ptr<tensor<8x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>) : i32 {
      %a = tt.load %subA {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x16xf16>, 1> -> tensor<8x16xf16>
      %b = tt.load %subB {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, DotB = true} : !tt.ptr<tensor<16x16xf16>, 1> -> tensor<16x16xf16>
      %c = tt.dot %a, %b, %arg4 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %30 = tt.advance %subA, [%c0, %c16] : <tensor<8x16xf16>, 1>
      %31 = tt.advance %subB, [%c16, %c0] : <tensor<16x16xf16>, 1>
      scf.yield %c, %30, %31 : tensor<8x16xf32>, !tt.ptr<tensor<8x16xf16>, 1>, !tt.ptr<tensor<16x16xf16>, 1>
    }
    %value = arith.truncf %6#0 : tensor<8x16xf32> to tensor<8x16xf16>
    %cPtr = tt.make_tensor_ptr %arg2, [%m, %n], [%n, %c1_i64], [%offsetYA, %offsetX] {order = array<i32: 1, 0>} : <tensor<8x16xf16>, 1>
    tt.store %cPtr, %value {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<8x16xf16>, 1>, tensor<8x16xf16>
    tt.return

}
}
