# DPAS Analysis Unit Tests

## Overview

This directory contains unit tests for the DPAS (Dot Product Accumulate Systolic) analysis component.

## Test Files

### DPASAnalysisTest.cpp

Comprehensive unit tests for the DPAS analysis infrastructure (`Analysis/DPAS.h` and `Analysis/DPAS.tpp`).

**Test Coverage:**

1. **DPAS Type Detection for DotOp:**
   - FP16 → FP32 (FP32_FP32_FP16_FP16)
   - BF16 → BF16 (BF16_BF16_BF16_BF16)
   - BF16 → FP32 (FP32_FP32_BF16_BF16)
   - FP32 TF32 → FP32 (FP32_FP32_TF32_TF32)
   - FP8 native support (FP32_FP32_FP8_FP8)
   - FP8 upcast to FP16 when not supported
   - INT8 signed (S32_S32_S8_S8)
   - INT8 unsigned (U32_U32_U8_U8)
   - Mismatched type handling
   - Xe3P-specific BF16_FP8 combinations

2. **DPAS Type Detection for DotScaledOp:**
   - BF16 × FP8 mixed precision
   - FP8 × FP8
   - FP4 × FP4 (with K-packing)
   - FP16 × FP8 mixed precision
   - Various scale element type combinations

3. **Function-Level Analysis (canUseDPAS):**
   - Empty functions
   - Valid DPAS operations
   - Wrong warp size configurations
   - Missing attributes (Maybe result)
   - Non-applicable DPAS types
   - Warp size 32 support

4. **DPASAnalysisFactory:**
   - Xe2 analysis creation
   - Xe3P analysis creation (with FP8 support)
   - Type variant handling
   - Operation-level analysis

5. **Edge Cases:**
   - No DPAS support
   - Multiple dot operations
   - Mixed valid/invalid operations

## Building and Running Tests

### Build Tests

From the build directory:

```bash
cmake --build . --target DPASAnalysis
```

Or build all unit tests:

```bash
cmake --build . --target check-intel-triton-unittest
```

### Run Tests

Run the DPAS analysis tests:

```bash
./unittest/DPASAnalysis
```

Run with verbose output:

```bash
./unittest/DPASAnalysis --gtest_print_time=1 --gtest_color=yes
```

Run specific test:

```bash
./unittest/DPASAnalysis --gtest_filter="DPASAnalysisTest.DotOp_FP16_FP16_FP32"
```

### Environment Variables

Some tests may behave differently based on environment variables:

- `TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32`: Enable DPAS for warp size 32

Example:

```bash
TRITON_INTEL_ENABLE_DPAS_FOR_WARP_SIZE_32=1 ./unittest/DPASAnalysis
```

## Adding New Tests

To add new test cases:

1. Add test methods to `DPASAnalysisTest` class
2. Use helper methods:
   - `createModule()`: Create a module with DPAS support flags
   - `createDotOp()`: Create a DotOp with specified types
   - `createDotScaledOp()`: Create a DotScaledOp with scales
   - `createFunction()`: Create a function in the module

Example:

```cpp
TEST_F(DPASAnalysisTest, MyNewTest) {
  auto module = createModule(/*supportDPAS=*/true);
  auto dotOp = createDotOp(module, builder->getF16Type(),
                           builder->getF16Type(), builder->getF32Type());

  auto dpasType = DPASAnalysisV1::getDPASType(dotOp);
  EXPECT_EQ(dpasType, DPASEngineTypeXe2::FP32_FP32_FP16_FP16);
}
```

## Test Infrastructure

- **Framework**: Google Test (gtest) and Google Mock (gmock)
- **MLIR Support**: Uses MLIR's C++ API for operation creation and manipulation
- **Isolation**: Each test creates its own MLIRContext and module

## Coverage Goals

Target coverage metrics:
- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Function Coverage**: 100% of public API

Current coverage can be measured using:
```bash
# Build with coverage flags
cmake -DLLVM_BUILD_INSTRUMENTED_COVERAGE=ON ...

# Run tests
./unittest/DPASAnalysis

# Generate coverage report
llvm-cov show ./unittest/DPASAnalysis -instr-profile=default.profdata
```
