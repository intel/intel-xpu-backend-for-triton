//===- Attributes.cpp - Construct MLIR attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "attributes"

using namespace llvm;
using namespace mlir;

namespace mlir::triton::gpu::intel {

constexpr StringLiteral AttributeList::PassthroughAttrName;

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

static void addToPassthroughAttr(NamedAttribute &PassthroughAttr,
                                 mlir::NamedAttribute Attr, MLIRContext &Ctx) {
  assert(PassthroughAttr.getName() == AttributeList::PassthroughAttrName &&
         "PassthroughAttr is not valid");
  assert(isa<ArrayAttr>(PassthroughAttr.getValue()) &&
         "PassthroughAttr should have an ArrayAttr as value");

  LLVM_DEBUG(llvm::dbgs() << "Adding attribute " << Attr.getName() << " to '"
                          << AttributeList::PassthroughAttrName << "'.\n";);

  std::vector<mlir::Attribute> Vec =
      cast<ArrayAttr>(PassthroughAttr.getValue()).getValue().vec();

  // TODO: find a way to add the attributes only if one does not exist already.
  if (isa<UnitAttr>(Attr.getValue()))
    Vec.push_back(Attr.getName());
  else
    Vec.push_back(ArrayAttr::get(&Ctx, {Attr.getName(), Attr.getValue()}));

  auto Comp = [&](const mlir::Attribute &A1, const mlir::Attribute &A2) {
    assert(isa<StringAttr>(A1) || isa<ArrayAttr>(A1));
    assert(isa<StringAttr>(A2) || isa<ArrayAttr>(A2));

    if (auto StrA1 = dyn_cast<StringAttr>(A1)) {
      if (auto StrA2 = dyn_cast<StringAttr>(A2))
        return StrA1 < StrA2;
      return true;
    }

    auto ArrA1 = cast<ArrayAttr>(A1);
    if (auto ArrA2 = dyn_cast<ArrayAttr>(A2))
      return cast<StringAttr>(ArrA1[0]) < cast<StringAttr>(ArrA2[0]);
    return false;
  };

  llvm::sort(Vec.begin(), Vec.end(), Comp);
  PassthroughAttr.setValue(ArrayAttr::get(&Ctx, Vec));

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << AttributeList::PassthroughAttrName << ": ( ";
    for (auto Item : Vec)
      llvm::dbgs() << Item << " ";
    llvm::dbgs() << ")\n";
  });
}

static void addToPassthroughAttr(mlir::NamedAttribute &PassthroughAttr,
                                 mlir::ArrayAttr NewAttributes,
                                 MLIRContext &Ctx) {
  assert(PassthroughAttr.getName() == AttributeList::PassthroughAttrName &&
         "PassthroughAttr is not valid");
  assert(isa<ArrayAttr>(PassthroughAttr.getValue()) &&
         "PassthroughAttr should have an ArrayAttr as value");

  for (mlir::Attribute NewAttr : NewAttributes) {
    if (auto ArrAttr = dyn_cast<ArrayAttr>(NewAttr)) {
      assert(ArrAttr.size() == 2 && isa<StringAttr>(ArrAttr[0]));
      NamedAttribute NamedAttr(cast<StringAttr>(ArrAttr[0]), ArrAttr[1]);
      addToPassthroughAttr(PassthroughAttr, NamedAttr, Ctx);
    } else if (auto StrAttr = dyn_cast<StringAttr>(NewAttr)) {
      NamedAttribute NamedAttr(StrAttr, UnitAttr::get(&Ctx));
      addToPassthroughAttr(PassthroughAttr, NamedAttr, Ctx);
    } else {
      llvm_unreachable("Unexpected attribute kind");
    }
  }
}

//===----------------------------------------------------------------------===//
// AttributeList Method Implementations
//===----------------------------------------------------------------------===//

AttributeList::AttributeList(
    const mlir::NamedAttrList &FnAttributes,
    const mlir::NamedAttrList &RetAttributes,
    llvm::ArrayRef<mlir::NamedAttrList> ParamAttributes)
    : FnAttributes(FnAttributes), RetAttributes(RetAttributes),
      ParamAttributes(ParamAttributes) {}

AttributeList &
AttributeList::addAttributes(const AttrBuilder &FnAttrB,
                             const AttrBuilder &RetAttrB,
                             llvm::ArrayRef<mlir::NamedAttrList> Attributes) {
  return addFnAttributes(FnAttrB).addRetAttributes(RetAttrB).addParamAttributes(
      Attributes);
}

AttributeList &AttributeList::addFnAttributes(const AttrBuilder &B) {
  return addFnAttributes(B.getAttributes(), B.getContext());
}

AttributeList &AttributeList::addFnAttributes(const NamedAttrList &Attributes,
                                              MLIRContext &Ctx) {
  for (const NamedAttribute &NewFnAttr : Attributes) {
    std::optional<NamedAttribute> ExistingFnAttr =
        FnAttributes.getNamed(NewFnAttr.getName());
    assert(
        (!ExistingFnAttr || ExistingFnAttr->getName() == PassthroughAttrName) &&
        "Function attribute already exists");
    if (!ExistingFnAttr) {
      FnAttributes.append(NewFnAttr);
    } else if (ExistingFnAttr->getName() == PassthroughAttrName) {
      // Merge the 'passthrough' attribute lists.
      auto Attributes = cast<ArrayAttr>(NewFnAttr.getValue());
      addToPassthroughAttr(*ExistingFnAttr, Attributes, Ctx);
      FnAttributes.set(ExistingFnAttr->getName(), ExistingFnAttr->getValue());
    }
  }

  return *this;
}

AttributeList &AttributeList::addRetAttributes(const AttrBuilder &B) {
  return addRetAttributes(B.getAttributes(), B.getContext());
}

AttributeList &
AttributeList::addRetAttributes(const mlir::NamedAttrList &Attributes,
                                mlir::MLIRContext &Ctx) {
  for (const NamedAttribute &NewNamedAttr : Attributes) {
    assert(RetAttributes.getNamed(NewNamedAttr.getName()).has_value() &&
           "Return value attribute already exists");
    RetAttributes.append(NewNamedAttr);
  }
  return *this;
}

AttributeList &AttributeList::addParamAttributes(
    llvm::ArrayRef<mlir::NamedAttrList> Attributes) {
  ParamAttributes.append(Attributes.begin(), Attributes.end());
  return *this;
}

//===----------------------------------------------------------------------===//
// AttrBuilder Method Implementations
//===----------------------------------------------------------------------===//

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind) {
  return addAttributeImpl(Kind, LLVM::LLVMDialect::getDialectNamespace(),
                          &AttrBuilder::addAttributeImpl);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
                                       mlir::Type Ty) {
  return addAttributeImpl(Kind, Ty, LLVM::LLVMDialect::getDialectNamespace(),
                          &AttrBuilder::addAttributeImpl);
}

AttrBuilder &AttrBuilder::addAttribute(llvm::Attribute::AttrKind Kind,
                                       uint64_t Val) {
  return addAttributeImpl(Kind, Val, &AttrBuilder::addRawIntAttr);
}

AttrBuilder &AttrBuilder::addAttribute(Twine AttrName, mlir::Attribute Attr) {
  return addAttributeImpl(AttrName, Attr, &AttrBuilder::addAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassthroughAttribute(llvm::Attribute::AttrKind Kind) {
  return addAttributeImpl(Kind, std::nullopt,
                          &AttrBuilder::addPassthroughAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassthroughAttribute(llvm::Attribute::AttrKind Kind,
                                     mlir::Type Ty) {
  return addAttributeImpl(Kind, Ty, std::nullopt,
                          &AttrBuilder::addPassthroughAttributeImpl);
}

AttrBuilder &
AttrBuilder::addPassthroughAttribute(llvm::Attribute::AttrKind Kind,
                                     uint64_t Val) {
  return addAttributeImpl(Kind, Val, &AttrBuilder::addPassthroughRawIntAttr);
}

AttrBuilder &AttrBuilder::addPassthroughAttribute(StringRef AttrName,
                                                  mlir::Attribute Attr) {
  return addAttributeImpl(AttrName, Attr,
                          &AttrBuilder::addPassthroughAttributeImpl);
}

AttrBuilder &AttrBuilder::removeAttribute(llvm::StringRef AttrName) {
  bool ContainsPassthroughAttr =
      getAttribute(AttributeList::PassthroughAttrName).has_value();
  if (ContainsPassthroughAttr) {
    NamedAttribute PassthroughAttr =
        getAttribute(AttributeList::PassthroughAttrName).value();
    auto ArrAttr = cast<ArrayAttr>(PassthroughAttr.getValue());
    std::vector<mlir::Attribute> Vec = ArrAttr.getValue().vec();

    llvm::remove_if(Vec, [AttrName](mlir::Attribute &Attr) {
      if (auto strAttr = dyn_cast<StringAttr>(Attr))
        return strAttr.strref() == AttrName;
      if (auto arrAttr = dyn_cast<ArrayAttr>(Attr)) {
        assert(arrAttr.size() == 2 && isa<StringAttr>(arrAttr[0]));
        return cast<StringAttr>(arrAttr[0]).strref() == AttrName;
      }
      return false;
    });

    PassthroughAttr.setValue(ArrayAttr::get(&Ctx, Vec));
    return *this;
  }

  Attributes.erase(AttrName);
  return *this;
}

AttrBuilder &AttrBuilder::removeAttribute(llvm::Attribute::AttrKind Kind) {
  assert(static_cast<unsigned>(Kind) < llvm::Attribute::EndAttrKinds &&
         "Attribute out of range!");
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return removeAttribute(AttrName);
}

bool AttrBuilder::contains(StringRef AttrName) const {
  return containsInPassthrough(AttrName) || getAttribute(AttrName).has_value();
}

bool AttrBuilder::contains(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return contains(AttrName);
}

std::optional<NamedAttribute>
AttrBuilder::getAttribute(StringRef AttrName) const {
  return Attributes.getNamed(AttrName);
}

std::optional<NamedAttribute>
AttrBuilder::getAttribute(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return getAttribute(AttrName);
}

StringAttr AttrBuilder::createStringAttribute(
    Twine AttrName, std::optional<StringLiteral> Prefix, MLIRContext &Ctx) {
  return (Prefix) ? StringAttr::get(&Ctx, *Prefix + "." + AttrName)
                  : StringAttr::get(&Ctx, AttrName);
}

AttrBuilder &AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind,
                                           std::optional<StringLiteral> Dialect,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");
  OpBuilder Builder(&Ctx);
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  NamedAttribute NamedAttr(createStringAttribute(AttrName, Dialect, Ctx),
                           Builder.getUnitAttr());
  return std::invoke(AddAttrPtr, this, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind,
                                           mlir::Type Ty,
                                           std::optional<StringLiteral> Dialect,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");
  OpBuilder Builder(&Ctx);
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  NamedAttribute NamedAttr(createStringAttribute(AttrName, Dialect, Ctx),
                           mlir::TypeAttr::get(Ty));
  return std::invoke(AddAttrPtr, this, NamedAttr);
}

AttrBuilder &
AttrBuilder::addAttributeImpl(llvm::Attribute::AttrKind Kind, uint64_t Val,
                              AddRawIntAttrFuncPtr AddRawIntAttrPtr) {
  assert(AddRawIntAttrPtr &&
         "'AddRawIntAttrPtr' should be a valid function pointer");

  switch (Kind) {
  case llvm::Attribute::AttrKind::Memory:
    // Val can be zero for memory(none).
    return std::invoke(AddRawIntAttrPtr, this, Kind, Val);
  case llvm::Attribute::AttrKind::Alignment:
    assert(Val <= llvm::Value::MaximumAlignment && "Alignment too large");
    return (!Val) ? *this : std::invoke(AddRawIntAttrPtr, this, Kind, Val);
  case llvm::Attribute::AttrKind::StackAlignment:
    assert(Val <= 0x100 && "Alignment too large.");
    LLVM_FALLTHROUGH;
  case llvm::Attribute::AttrKind::Dereferenceable:
  case llvm::Attribute::AttrKind::DereferenceableOrNull:
  case llvm::Attribute::AttrKind::UWTable:
    return (!Val) ? *this : std::invoke(AddRawIntAttrPtr, this, Kind, Val);

  default:
    llvm_unreachable("Unexpected attribute kind");
  }

  return *this;
}

AttrBuilder &AttrBuilder::addAttributeImpl(Twine AttrName, mlir::Attribute Attr,
                                           AddAttrFuncPtr AddAttrPtr) {
  assert(AddAttrPtr && "'AddAttrPtr' should be a valid function pointer");
  NamedAttribute NamedAttr(StringAttr::get(&Ctx, AttrName), Attr);
  return std::invoke(AddAttrPtr, this, NamedAttr);
}

AttrBuilder &AttrBuilder::addAttributeImpl(mlir::NamedAttribute Attr) {
  Attributes.set(Attr.getName(), Attr.getValue());
  return *this;
}

AttrBuilder &
AttrBuilder::addPassthroughAttributeImpl(mlir::NamedAttribute Attr) {
  NamedAttribute PassthroughAttr = getOrCreatePassthroughAttr();
  addToPassthroughAttr(PassthroughAttr, Attr, Ctx);
  return addAttributeImpl(PassthroughAttr);
}

AttrBuilder &AttrBuilder::addRawIntAttr(llvm::Attribute::AttrKind Kind,
                                        uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr(
      createStringAttribute(llvm::Attribute::getNameFromAttrKind(Kind),
                            LLVM::LLVMDialect::getDialectNamespace(), Ctx),
      Builder.getIntegerAttr(Builder.getIntegerType(64), Value));
  return addAttributeImpl(NamedAttr);
}

AttrBuilder &
AttrBuilder::addPassthroughRawIntAttr(llvm::Attribute::AttrKind Kind,
                                      uint64_t Value) {
  OpBuilder Builder(&Ctx);
  NamedAttribute NamedAttr(
      StringAttr::get(&Ctx, llvm::Attribute::getNameFromAttrKind(Kind)),
      StringAttr::get(&Ctx, Twine(Value)));
  return addPassthroughAttributeImpl(NamedAttr);
}

NamedAttribute AttrBuilder::getOrCreatePassthroughAttr() const {
  std::optional<NamedAttribute> PassthroughAttr =
      getAttribute(AttributeList::PassthroughAttrName);
  if (!PassthroughAttr) {
    LLVM_DEBUG(llvm::dbgs()
               << "Creating empty '" << AttributeList::PassthroughAttrName
               << "' attribute\n");
    PassthroughAttr = NamedAttribute(
        StringAttr::get(&Ctx, AttributeList::PassthroughAttrName),
        ArrayAttr::get(&Ctx, {}));
  }
  return *PassthroughAttr;
}

bool AttrBuilder::containsInPassthrough(StringRef AttrName) const {
  if (!getAttribute(AttributeList::PassthroughAttrName).has_value())
    return false;

  NamedAttribute PassthroughAttr =
      getAttribute(AttributeList::PassthroughAttrName).value();
  assert(isa<ArrayAttr>(PassthroughAttr.getValue()) &&
         "passthrough attribute value should be an ArrayAttr");

  return llvm::any_of(
      cast<ArrayAttr>(PassthroughAttr.getValue()),
      [AttrName](mlir::Attribute Attr) {
        if (isa<ArrayAttr>(Attr)) {
          auto ArrAttr = cast<ArrayAttr>(Attr);
          assert(ArrAttr.size() == 2 && isa<StringAttr>(ArrAttr[0]));
          return cast<StringAttr>(ArrAttr[0]) == AttrName;
        }

        assert(isa<StringAttr>(Attr) && "Unexpected attribute Kind");
        return cast<StringAttr>(Attr) == AttrName;
      });
}

bool AttrBuilder::containsInPassthrough(llvm::Attribute::AttrKind Kind) const {
  StringRef AttrName = llvm::Attribute::getNameFromAttrKind(Kind);
  return containsInPassthrough(AttrName);
}

} // namespace mlir::triton::gpu::intel
