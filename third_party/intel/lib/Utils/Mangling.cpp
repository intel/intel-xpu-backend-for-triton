#include "Mangling.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::gpu::intel {
std::string getTypeMangling(Type ty, bool isUnsigned) {
  return TypeSwitch<Type, std::string>(ty)
      .Case([isUnsigned](VectorType ty) -> std::string {
        return "Dv" + std::to_string(ty.getNumElements()) + "_" +
               getTypeMangling(ty.getElementType(), isUnsigned);
      })
      .Case([](Float16Type) -> std::string { return "Dh"; })
      .Case([](Float32Type) -> std::string { return "f"; })
      .Case([](Float64Type) -> std::string { return "d"; })
      .Case([isUnsigned](IntegerType ty) -> std::string {
        switch (ty.getWidth()) {
        case 8:
          return isUnsigned ? "h" : "c";
        case 16:
          return isUnsigned ? "t" : "s";
        case 32:
          return isUnsigned ? "j" : "i";
        case 64:
          return isUnsigned ? "m" : "l";
        default:
          llvm_unreachable("unhandled integer type");
        }
      });
}

std::string mangle(StringRef baseName, ArrayRef<Type> types,
                   ArrayRef<bool> isUnsigned) {
  assert((isUnsigned.empty() || isUnsigned.size() == types.size()) &&
         "Signedness info doesn't match");
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::SmallDenseMap<Type, unsigned> substitutions;
  os << "_Z" << baseName.size() << baseName;
  for (auto [idx, type] : llvm::enumerate(types)) {
    auto it = substitutions.find(type);
    if (it != substitutions.end()) {
      os << "S";
      // First substitution is `S_`, second is `S0_`, and so on.
      if (unsigned firstIdx = it->getSecond(); firstIdx > 0)
        os << firstIdx - 1;
      os << "_";
    } else {
      if (!type.isIntOrFloat())
        substitutions[type] = substitutions.size();
      os << getTypeMangling(type, isUnsigned.empty() ? false : isUnsigned[idx]);
    }
  }
  return os.str();
}
} // namespace mlir::triton::gpu::intel
