// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7336
// COM: A value feeding both a tt.scan and a convert_layout to a non-blocked
// COM: (slice) encoding used to crash backward rematerialization: the layout
// COM: was forward-propagated onto the scan, inferDstEncoding returned a null
// COM: Attribute for the scan, and the null later hit a dyn_cast (isPresent
// COM: assertion / segfault). The pass must now complete and keep the scan.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func @scan_forward_remat_slice
  tt.func @scan_forward_remat_slice() -> (tensor<16xi32, #blocked>, tensor<16xi32, #slice>) {
    %a = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #blocked>
    %add = arith.addi %a, %a : tensor<16xi32, #blocked>
    // CHECK: "tt.scan"
    %scan = "tt.scan"(%add) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      tt.scan.return %sum : i32
    }) : (tensor<16xi32, #blocked>) -> tensor<16xi32, #blocked>
    %cvt = ttg.convert_layout %add : tensor<16xi32, #blocked> -> tensor<16xi32, #slice>
    // CHECK: tt.return
    tt.return %scan, %cvt : tensor<16xi32, #blocked>, tensor<16xi32, #slice>
  }
}
