#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <set>

#define DEBUG_TYPE "ttgpu_to_llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define inttoptr(...) rewriter.create<LLVM::IntToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<LLVM::PtrToIntOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<LLVM::ZExtOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<LLVM::SExtOp>(loc, __VA_ARGS__)
#define fpext(...) rewriter.create<LLVM::FPExtOp>(loc, __VA_ARGS__)
#define trunc(...) rewriter.create<LLVM::TruncOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<LLVM::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<LLVM::URemOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<LLVM::AddOp>(loc, __VA_ARGS__)
// #define sub(...) rewriter.create<LLVM::SubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<LLVM::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<LLVM::MulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<LLVM::FMulOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<LLVM::SMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<LLVM::UMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<LLVM::MaxNumOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<LLVM::SMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<LLVM::UMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<LLVM::MinNumOp>(loc, __VA_ARGS__)
#define shl(...) rewriter.create<LLVM::ShlOp>(loc, __VA_ARGS__)
#define lshr(...) rewriter.create<LLVM::LShrOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<LLVM::AndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<LLVM::XOrOp>(loc, __VA_ARGS__)
#define or_(...) rewriter.create<LLVM::OrOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)                                                 \
  rewriter.create<LLVM::BitcastOp>(loc, type__, val__)
#define gep(...) rewriter.create<LLVM::GEPOp>(loc, __VA_ARGS__)
#define ptr_ty(...) LLVM::LLVMPointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<LLVM::InsertValueOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<LLVM::ExtractValueOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<LLVM::InsertElementOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<LLVM::ExtractElementOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<LLVM::LoadOp>(loc, __VA_ARGS__)
// #define store(val, ptr) rewriter.create<LLVM::StoreOp>(loc, val, ptr)
// #define load_dsmem(...) LLVM::createLoadDSmem(loc, rewriter, __VA_ARGS__)
// #define store_dsmem(...) LLVM::createStoreDSmem(loc, rewriter, __VA_ARGS__)
// #define addrspacecast(val__, type__)                                           \
//   rewriter.create<LLVM::AddrSpaceCastOp>(loc, type__, val__)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::ogt, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::olt, lhs, rhs)
#define fcmp_eq(lhs, rhs)                                                      \
  rewriter.create<LLVM::FCmpOp>(loc, rewriter.getI1Type(),                     \
                                LLVM::FCmpPredicate::oeq, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, __VA_ARGS__)
#define select(...) rewriter.create<LLVM::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<LLVM::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<LLVM::UndefOp>(loc, __VA_ARGS__)
#define null(...) rewriter.create<LLVM::ZeroOp>(loc, __VA_ARGS__)
// #define call(...) rewriter.create<LLVM::CallOp>(loc, __VA_ARGS__)

// Types
#define int_ty(width) rewriter.getIntegerType(width)
#define i64_ty rewriter.getIntegerType(64)
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define i32_ty rewriter.getIntegerType(32)
#define i64_ty rewriter.getIntegerType(64)
#define ui32_ty rewriter.getIntegerType(32, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define i1_ty rewriter.getI1Type()
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define void_ty(ctx) LLVM::LLVMVoidType::get(ctx)
#define struct_ty(...) LLVM::LLVMStructType::getLiteral(ctx, __VA_ARGS__)
#define array_ty(elemTy, count) LLVM::LLVMArrayType::get(elemTy, count)

// Constants
#define f32_val(...) LLVM::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) LLVM::createConstantF64(loc, rewriter, __VA_ARGS__)
#define i32_val(...) LLVM::createConstantI32(loc, rewriter, __VA_ARGS__)
#define int_val(width, val)                                                    \
  LLVM::createLLVMIntegerConstant(rewriter, loc, width, val)
#define tid_val() getThreadId(rewriter, loc)

// Attributes
#define i32_arr_attr(...) rewriter.getI32ArrayAttr({__VA_ARGS__})
#define i64_arr_attr(...) rewriter.getI64ArrayAttr({__VA_ARGS__})
#endif

namespace mlir {
namespace triton {

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
                                          llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
                                      llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
                 llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(applyPermutation(multiDimIndex, order),
                               applyPermutation(shape, order));
}

} // namespace triton

/* ------------------------------------ */
// Returns CTA level thread idx
static Value getThreadIdInCTA(RewriterBase &rewriter, Location loc) {
  Value tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);
}

// Returns CTA level thread idx.
static Value getThreadId(RewriterBase &rewriter, Location loc) {
  Value tid = getThreadIdInCTA(rewriter, loc);
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return tid;
}

// static Value getClusterCTAId(RewriterBase &rewriter, Location loc) {
//   return rewriter.create<triton::nvgpu::ClusterCTAIdOp>(loc,
//                                                         rewriter.getI32Type());
// }

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
// using LLVM::getMultiDimIndex;
// using LLVM::SharedMemoryObject;
// using ::mlir::LLVM::delinearize;
// using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

// static Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
//                  ArrayRef<Value> strides) {
//   assert(offsets.size() == strides.size());
//   Value ret = LLVM::createConstantI32(loc, rewriter, 0);
//   for (auto [offset, stride] : llvm::zip(offsets, strides)) {
//     ret = add(ret, mul(offset, stride));
//   }
//   return ret;
// }



} // namespace mlir

