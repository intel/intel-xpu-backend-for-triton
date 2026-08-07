//===- ArchParserTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for parse_device_arch() in arch_parser.c, which maps SYCL device
// architecture enum values to the architecture strings used by the Intel XPU
// backend. The version-gated cases mirror the #if guards in arch_parser.c so
// the test list automatically tracks what the SYCL toolchain supports.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <string>

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

namespace arch = sycl::ext::oneapi::experimental;

// The production function under test, defined in arch_parser.c.
extern "C" const char *parse_device_arch(uint64_t dev_arch);

namespace {

// Invokes the production parser with a SYCL architecture enum value.
std::string parse(arch::architecture a) {
  return parse_device_arch(static_cast<uint64_t>(a));
}

struct ArchCase {
  arch::architecture arch;
  const char *expected;
};

class ArchParserKnownArch : public ::testing::TestWithParam<ArchCase> {};

TEST_P(ArchParserKnownArch, ReturnsExpectedString) {
  const ArchCase &c = GetParam();
  EXPECT_EQ(parse(c.arch), c.expected);
}

// Each known architecture maps to its expected string. The guards match those
// in arch_parser.c: a case is only tested when the SYCL compiler defines it.
const ArchCase kKnownArchitectures[] = {
    {arch::architecture::intel_gpu_arl_h, "arl_h"},
    {arch::architecture::intel_gpu_arl_s, "arl_s"},
    {arch::architecture::intel_gpu_bmg_g21, "bmg"},
#if __SYCL_COMPILER_VERSION >= 20251010
    {arch::architecture::intel_gpu_bmg_g31, "bmg"},
#endif
    {arch::architecture::intel_gpu_dg2_g10, "dg2"},
    {arch::architecture::intel_gpu_lnl_m, "lnl"},
    {arch::architecture::intel_gpu_mtl_h, "mtl"},
    {arch::architecture::intel_gpu_pvc, "pvc"},
#if __SYCL_COMPILER_VERSION >= 20250000
    {arch::architecture::intel_gpu_ptl_h, "ptl_h"},
    {arch::architecture::intel_gpu_ptl_u, "ptl_u"},
#endif
#if __SYCL_COMPILER_VERSION >= 20260331
    {arch::architecture::intel_gpu_nvl_s, "nvl_s"},
    {arch::architecture::intel_gpu_nvl_u, "nvl_u"},
    {arch::architecture::intel_gpu_nvl_p, "nvl_p"},
#endif
};

INSTANTIATE_TEST_SUITE_P(KnownArchitectures, ArchParserKnownArch,
                         ::testing::ValuesIn(kKnownArchitectures));

// The `unknown` enum is an explicit case in the switch and maps to "unknown",
// distinct from an unrecognized value (see below).
TEST(ArchParser, UnknownEnumReturnsUnknownString) {
  EXPECT_EQ(parse(arch::architecture::unknown), "unknown");
}

// A value matching no enum case falls through to `default` and returns the
// empty string (the function also logs to stderr in this path).
TEST(ArchParser, UnrecognizedValueReturnsEmptyString) {
  // A large value that does not correspond to any SYCL architecture enumerator.
  constexpr uint64_t kUnrecognized = 0xFFFFFFFFFFFFFFFFULL;
  EXPECT_EQ(std::string(parse_device_arch(kUnrecognized)), "");
}

} // namespace
