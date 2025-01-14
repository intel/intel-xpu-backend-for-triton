#ifndef PYBIND_TYPE_CASTERS_H
#define PYBIND_TYPE_CASTERS_H

#include "llvm/ADT/SmallVector.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace pybind11 {
namespace detail {

////////////////////////////////////////////////////////////////////////////////
/// LLVM::SMALLVECTOR BINDING
////////////////////////////////////////////////////////////////////////////////

template <> struct type_caster<llvm::SmallVector<std::string>> {
  PYBIND11_TYPE_CASTER(llvm::SmallVector<std::string>, _("List[str]"));

  bool load(handle src, bool) {
    if (!pybind11::isinstance<sequence>(src)) {
      return false;
    }

    auto vec = pybind11::reinterpret_borrow<sequence>(src);
    llvm::SmallVector<std::string> tmp;
    for (auto s : vec) {
      if (!pybind11::isinstance<str>(s)) {
        return false;
      }
      tmp.push_back(pybind11::cast<std::string>(s));
    }
    value = std::move(tmp);

    return true;
  }
};

} // namespace detail
} // namespace pybind11

#endif // PYBIND_TYPE_CASTERS_H
