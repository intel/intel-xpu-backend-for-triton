#include "intel/include/TritonIntelGPUToLLVM/XeAsmFormat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Signals.h"

#include <gtest/gtest.h>

namespace mlir {
namespace triton {
class XeAsmFormatTest : public ::testing::Test {
protected:
  static constexpr int numValues = 4;

  XeAsmFormatTest() {
    ctx.loadDialect<arith::ArithDialect>();

    createValues();
  }

  // Creates the test values.
  void createValues() {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&block);

    // a b1 value for predicate.
    v[0] = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
    for (int i = 0; i < numValues; i++) {
      v[i + 1] =
          builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), i, 32);
    }
  }

  MLIRContext ctx;
  Block block;
  Value v[numValues + 1];
};

TEST_F(XeAsmFormatTest, VISA_basic) {
  XeBuilder builder;

  // Create the operands needed by the instructions in the Xe code.
  auto *cst = builder.newConstantOperand(1);
  auto *val = builder.newOperand(v[0], "=r");

  // create an instruction
  auto &mov = *builder.create("mov (M1_NM, 1)");

  mov(val, cst);
  ASSERT_EQ(builder.dump(), "mov (M1_NM, 1) $0, 0x1;");

  auto values = builder.getAllMLIRArgs();
  ASSERT_EQ(values[0], v[0]); // $0 -> v[1]

  auto constraints = builder.getConstraints();
  ASSERT_EQ(constraints, "=r"); // $0 -> =r
}

TEST_F(XeAsmFormatTest, MultiLineXe) {
  XeBuilder builder;

  auto *constVal = builder.newConstantOperand(1);
  auto *valVal0 = builder.newOperand(v[1], "=r");
  auto *valVal1 = builder.newOperand(v[2], "=r");

  auto &mov = *builder.create("mov");

  mov(valVal0, constVal);
  mov(valVal1, constVal);
  mov(valVal1, valVal0);

  EXPECT_EQ(builder.dump(), "mov $0, 0x1;\n\t"
                            "mov $1, 0x1;\n\t"
                            "mov $1, $0;");

  auto values = builder.getAllMLIRArgs();
  EXPECT_EQ(values[0], v[1]); // $0 -> v[1]
  EXPECT_EQ(values[1], v[2]); // $1 -> v[2]
}

TEST_F(XeAsmFormatTest, XeSIMDReduce) {
  auto assemble = mlir::triton::simdReduceAsm(
      "add", 16, 16, 16, Float16Type::get(&ctx), XeArch::Xe);
  llvm::outs() << "johnlu:" << assemble;
  assemble = mlir::triton::simdReduceAsm("max", 16, 16, 16,
                                         Float32Type::get(&ctx), XeArch::Xe);
  llvm::outs() << "johnlu:" << assemble;
  assemble = mlir::triton::simdReduceAsm("add", 32, 32, 32,
                                         Float32Type::get(&ctx), XeArch::Xe2);
  llvm::outs() << "johnlu:" << assemble;
  assemble = mlir::triton::simdReduceAsm("max", 32, 32, 32,
                                         Float32Type::get(&ctx), XeArch::Xe2);
  llvm::outs() << "johnlu:" << assemble;
  // mlir::triton::simdReduceAsm("max", 16, 16, IntegerType::get(&ctx, 32));
}

} // namespace triton
} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
