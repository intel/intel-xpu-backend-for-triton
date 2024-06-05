//===- Attributes.h - Construct MLIR attributes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ATTRIBUTES_H
#define TRITON_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/IR/Attributes.h"
#include <optional>

namespace mlir::triton::gpu::intel {

class AttrBuilder;

/// \class
/// This class holds the attributes for a function, its return value, and
/// its parameters.
class AttributeList {
public:
  static constexpr llvm::StringLiteral PassthroughAttrName = "passthrough";

  AttributeList() = default;
  AttributeList(const mlir::NamedAttrList &FnAttributes,
                const mlir::NamedAttrList &RetAttributes,
                llvm::ArrayRef<mlir::NamedAttrList> ParamAttributes);

  //===--------------------------------------------------------------------===//
  // AttributeList Mutation
  //===--------------------------------------------------------------------===//

  /// Add function, return value, and parameters attributes to the list.
  AttributeList &addAttributes(const AttrBuilder &FnAttrB,
                               const AttrBuilder &RetAttrB,
                               llvm::ArrayRef<mlir::NamedAttrList> Attributes);

  /// Add function attributes to the list.
  AttributeList &addFnAttributes(const AttrBuilder &B);
  AttributeList &addFnAttributes(const mlir::NamedAttrList &Attributes,
                                 mlir::MLIRContext &Ctx);

  /// Add return value attributes to the list.
  AttributeList &addRetAttributes(const AttrBuilder &B);
  AttributeList &addRetAttributes(const mlir::NamedAttrList &Attributes,
                                  mlir::MLIRContext &Ctx);

  /// Add parameters attributes to the list.
  AttributeList &
  addParamAttributes(llvm::ArrayRef<mlir::NamedAttrList> Attributes);

  /// The function attributes are returned.
  const mlir::NamedAttrList &getFnAttributes() const { return FnAttributes; }

  /// The attributes for the ret value are returned.
  const mlir::NamedAttrList &getRetAttributes() const { return RetAttributes; }

  /// The attributes for the parameters are returned.
  const mlir::ArrayRef<mlir::NamedAttrList> getParamAttributes() const {
    return ParamAttributes;
  }

private:
  /// The attributes that we are managing.
  mlir::NamedAttrList FnAttributes;
  mlir::NamedAttrList RetAttributes;
  llvm::SmallVector<mlir::NamedAttrList, 8> ParamAttributes;
};

/// \class
/// Facilitates the construction of LLVM dialect attributes for a particular
/// argument, parameter, function, or return value.
class AttrBuilder {
public:
  AttrBuilder(mlir::MLIRContext &Ctx) : Ctx(Ctx) {}
  AttrBuilder(const AttrBuilder &) = delete;
  AttrBuilder(AttrBuilder &&) = default;

  /// Add the LLVM attribute identified by \p Kind to the builder.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind Kind);

  /// Add the LLVM attribute identified by \p Kind with a type given by \p Ty
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind Kind, mlir::Type Ty);

  /// Add the LLVM attribute identified by \p Kind with a value given by \p Val
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Attribute::AttrKind Kind, uint64_t Val);

  /// Create a NamedAttribute with name \p AttrName and value \p Attr and add it
  /// to the builder.
  AttrBuilder &addAttribute(llvm::Twine AttrName, mlir::Attribute Attr);

  /// Add the LLVM attribute identified by \p Kind to the builder "passthrough"
  /// named attribute.
  AttrBuilder &addPassthroughAttribute(llvm::Attribute::AttrKind Kind);

  /// Add the LLVM attribute identified by \p Kind with a type given by \p Ty
  /// to the builder "passthrough" named attribute.
  AttrBuilder &addPassthroughAttribute(llvm::Attribute::AttrKind Kind,
                                       mlir::Type Ty);

  /// Add the LLVM attribute identified by \p Kind with a value given by \p Val
  /// to the builder "passthrough" named attribute.
  AttrBuilder &addPassthroughAttribute(llvm::Attribute::AttrKind Kind,
                                       uint64_t Val);

  /// Create a NamedAttribute with name \p AttrName and value \p Attr and add it
  /// to the builder "passthrough" named attribute.
  AttrBuilder &addPassthroughAttribute(llvm::StringRef AttrName,
                                       mlir::Attribute Attr);

  /// Remove an attribute from the builder (if present).
  /// Note: the given attribute will be removed even if it is contained by the
  /// 'passthrough' named attribute.
  AttrBuilder &removeAttribute(llvm::StringRef AttrName);
  AttrBuilder &removeAttribute(llvm::Attribute::AttrKind Kind);

  /// Return true if the builder contains the specified attribute.
  /// Note: these member functions also lookup for the given attribute in the
  /// 'passthrough' named attribute if it exists.
  bool contains(llvm::StringRef AttrName) const;
  bool contains(llvm::Attribute::AttrKind Kind) const;

  /// Return true if the builder contains any attribute and false otherwise.
  bool hasAttributes() const { return !Attributes.empty(); }

  /// Return the given attribute if the builder contains it and llvm::None
  /// otherwise.
  std::optional<mlir::NamedAttribute>
  getAttribute(llvm::StringRef AttrName) const;
  std::optional<mlir::NamedAttribute>
  getAttribute(llvm::Attribute::AttrKind Kind) const;

  /// Returns the attributes contained in the builder.
  llvm::ArrayRef<mlir::NamedAttribute> getAttributes() const {
    return Attributes;
  }

  mlir::MLIRContext &getContext() const { return Ctx; }

  /// Returns a StringAttr of the form 'prefix.AttrName'.
  static mlir::StringAttr
  createStringAttribute(llvm::Twine AttrName,
                        std::optional<llvm::StringLiteral> Prefix,
                        mlir::MLIRContext &Ctx);

private:
  using AddAttrFuncPtr =
      AttrBuilder &(AttrBuilder::*)(mlir::NamedAttribute Attr);
  using AddRawIntAttrFuncPtr = AttrBuilder &(
      AttrBuilder::*)(llvm::Attribute::AttrKind Kind, uint64_t Value);

  /// Add the LLVM attribute identified by \p Kind to the builder, optionally
  /// prefixing the attribute name with \p Dialect.
  /// Note: \p AddAttrPtr is used to provide a concrete implementation
  /// controlling where to add the attribute (to the 'passthrough' list or not).
  AttrBuilder &addAttributeImpl(llvm::Attribute::AttrKind Kind,
                                std::optional<llvm::StringLiteral> Dialect,
                                AddAttrFuncPtr AddAttrPtr);

  /// Add the LLVM attribute identified by \p Kind with a type given by \p Ty
  /// to the builder, optionally prefixing the attribute name with \p Dialect.
  /// Note: \p AddAttrPtr is used to provide a concrete implementation
  /// controlling where to add the attribute (to the 'passthrough' list or not).
  AttrBuilder &addAttributeImpl(llvm::Attribute::AttrKind Kind, mlir::Type Ty,
                                std::optional<llvm::StringLiteral> Dialect,
                                AddAttrFuncPtr AddAttrPtr);

  /// Add the LLVM attribute identified by \p Kind with a value given by \p Val
  /// to the builder.
  /// Note: \p AddRawIntAttrPtr is used to provide a concrete implementation
  /// controlling where to add the attribute (to the 'passthrough' list or not).
  AttrBuilder &addAttributeImpl(llvm::Attribute::AttrKind Kind, uint64_t Val,
                                AddRawIntAttrFuncPtr AddRawIntAttrPtr);

  /// Create a NamedAttribute with name \p AttrName and value \p Attr and add it
  /// to the builder.
  /// Note: \p AddAttrPtr is used to provide a concrete implementation
  /// controlling where to add the attribute (to the 'passthrough' list or not).
  AttrBuilder &addAttributeImpl(llvm::Twine AttrName, mlir::Attribute Attr,
                                AddAttrFuncPtr AddAttrPtr);

  /// Add the given named attribute \p Attr to the builder.
  AttrBuilder &addAttributeImpl(mlir::NamedAttribute Attr);

  /// Add the given named attribute \p Attr to the builder "passthrough" named
  /// attribute.
  AttrBuilder &addPassthroughAttributeImpl(mlir::NamedAttribute Attr);

  /// Add integer attribute with raw value (packed/encoded if necessary).
  AttrBuilder &addRawIntAttr(llvm::Attribute::AttrKind Kind, uint64_t Value);

  /// Add integer attribute with raw value (packed/encoded if necessary) to the
  /// builder "passthrough" named attribute.
  AttrBuilder &addPassthroughRawIntAttr(llvm::Attribute::AttrKind Kind,
                                        uint64_t Value);

  /// Retrieve the "passthrough" named attribute if present, create it with an
  /// empty list otherwise.
  mlir::NamedAttribute getOrCreatePassthroughAttr() const;

  /// Return true if the builder contains the specified attribute within the
  /// 'passthrough' attribute.
  bool containsInPassthrough(llvm::StringRef AttrName) const;
  bool containsInPassthrough(llvm::Attribute::AttrKind Kind) const;

  mlir::MLIRContext &Ctx;
  mlir::NamedAttrList Attributes;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_ATTRIBUTES_H
