#include "Mangling.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {
std::string getTypeMangling(Type ty) {
  return TypeSwitch<Type, std::string>(ty)
      .Case([](VectorType ty) -> std::string {
        return "Dv" + std::to_string(ty.getNumElements()) + "_" +
               getTypeMangling(ty.getElementType());
      })
      .Case([](Float16Type) -> std::string { return "Dh"; })
      .Case([](Float32Type) -> std::string { return "f"; })
      .Case([](Float64Type) -> std::string { return "d"; })
      .Case([](IntegerType ty) -> std::string {
        switch (ty.getWidth()) {
        case 8:
          return "c";
        case 16:
          return "s";
        case 32:
          return "i";
        case 64:
          return "l";
        default:
          llvm_unreachable("unhandled integer type");
        }
      });
}

std::string mangle(StringRef baseName, ArrayRef<Type> types) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "_Z" << baseName.size() << baseName;
  for (Type type : types)
    os << getTypeMangling(type);
  return os.str();
}
} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
