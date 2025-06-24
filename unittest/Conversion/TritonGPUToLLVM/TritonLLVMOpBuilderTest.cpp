#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonLLVMOpBuilderTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  void SetUp() override {
    ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx.getOrLoadDialect<mlir::triton::TritonDialect>();
  }
};

TEST_F(TritonLLVMOpBuilderTest, FillTensorWithConstants) {
  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);

  mlir::PassManager pm(&ctx);
  //   pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Create tensor<1024x1024xi32> type
  auto elemType = builder.getIntegerType(32);
  auto tensorType = RankedTensorType::get({1024, 1024}, elemType);

  // Create function type: () -> tensor<1024x1024xi32>
  auto funcType = builder.getFunctionType({}, {tensorType});
  auto func = builder.create<mlir::triton::FuncOp>(loc, "test", funcType);
  module.push_back(func);
  // auto &entryBlock = *func.addEntryBlock();
  builder.setInsertionPointToStart(func.addEntryBlock());

  Value tensor = builder.create<mlir::tensor::EmptyOp>(
      loc, ArrayRef<int64_t>({1024, 1024}), elemType);
  auto term = builder.create<mlir::triton::ReturnOp>(loc, tensor);

  TritonLLVMOpBuilder ttb(loc, builder);
  ttb.i32_val(0);
  ttb.i32_val(0);
  ttb.i32_val(1);
  ttb.i32_val(2);
  SmallVector<Attribute, 16> values;
  for (double v : {0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2.,
                   -3., -4., -6.})
    values.push_back(builder.getFloatAttr(builder.getBF16Type(), v));
  ttb.dense_val(VectorType::get({16}, builder.getBF16Type()), values);

  module.dump();
  if (failed(pm.run(module))) {
    std::cerr << "CSE pass failed!" << std::endl;
  }
  module.dump();
  TritonLLVMOpBuilder::clear_cache(module.getBody());
  module.dump();
  term.erase();

  auto loopStart = std::chrono::high_resolution_clock::now();
  for (int row = 0; row < 1024; ++row) {
    for (int col = 0; col < 1024; ++col) {
      Value val = ttb.i32_val(col);
      Value rowIdx = ttb.idx_val(row);
      Value colIdx = ttb.idx_val(col);
      SmallVector<Value, 2> indices{rowIdx, colIdx};
      tensor =
          builder.create<mlir::tensor::InsertOp>(loc, val, tensor, indices);
    }
  }
  auto loopEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> loopDuration = loopEnd - loopStart;
  std::cout << "Nested loops execution time: " << loopDuration.count()
            << " seconds" << std::endl;

  builder.create<mlir::triton::ReturnOp>(loc, tensor);
  // module.push_back(func);
  // module.dump();
  ASSERT_TRUE(succeeded(mlir::verify(module)));

  auto cseStart = std::chrono::high_resolution_clock::now();
  TritonLLVMOpBuilder::clear_cache(module.getBody());
  if (failed(pm.run(module))) {
    std::cerr << "CSE pass failed!" << std::endl;
  }
  auto cseEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cseDuration = cseEnd - cseStart;
  std::cout << "CSE pass execution time: " << cseDuration.count() << " seconds"
            << std::endl;

  // Dump the final module
  //   module.dump();

  // Optionally, verify the module
  //   ASSERT_TRUE(succeeded(mlir::verify(module)));
}

} // namespace
