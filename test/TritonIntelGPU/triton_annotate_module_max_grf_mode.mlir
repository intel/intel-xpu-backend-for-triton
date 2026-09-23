// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16 max-grf-mode=256' | FileCheck %s --check-prefix=CHECK-SET
// RUN: triton-opt %s --split-input-file -triton-annotate-module='min-sg-size=16' | FileCheck %s --check-prefix=CHECK-UNSET

// COM: Dedicated coverage for the 'max-grf-mode' option (see issue #8074):
// COM: RegisterPressureAnalysis reads the 'ttig.max_grf_mode' module attribute this
// COM: stamps to resolve UnknownGRFSizeAssumption::Largest for "default"/"auto" GRF modes.

module {
  // COM: Ensure 'max-grf-mode' is stamped as the 'ttig.max_grf_mode' module attribute
  //      when the option is explicitly set.
  // CHECK-SET: module attributes {{.*}}ttig.max_grf_mode = "256"{{.*}}
  // COM: The default (unset) case must leave the attribute unstamped, not stamp some
  //      fallback value -- callers must not assume the attribute is always present, and
  //      RegisterPressureAnalysis's own fallback owns that case instead.
  // CHECK-UNSET-NOT: ttig.max_grf_mode

  tt.func @kernel() {
    tt.return
  }
}
