// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 grf-register-size=64' | FileCheck %s --check-prefix=CHECK-SET
// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16' | FileCheck %s --check-prefix=CHECK-UNSET

// COM: Dedicated coverage for the 'grf-register-size' option: RegisterPressureAnalysis
// COM: reads the 'ttig.grf_register_size' module attribute this stamps to turn a GRF
// COM: mode's register count into a byte budget.

module {
  // COM: Ensure 'grf-register-size' is stamped as the 'ttig.grf_register_size' module
  //      attribute when the option is explicitly set.
  // CHECK-SET: module attributes {{.*}}ttig.grf_register_size = 64 : i32{{.*}}
  // COM: The default (unset) case must leave the attribute unstamped, not stamp some
  //      fallback value: RegisterPressureAnalysis's own fallback owns that case instead.
  // CHECK-UNSET-NOT: ttig.grf_register_size

  tt.func @kernel() {
    tt.return
  }
}
