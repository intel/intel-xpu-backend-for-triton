---
description: 'AI-assisted coding guidelines following LLVM/MLIR standards'
applyTo: '**/*.cpp, **/*.tpp, **/*.c, **/*.h, **/*.hpp'
---

# LLVM/MLIR Coding Guidelines for AI Assistance

## Overview
Generate code following LLVM/MLIR standards. Apply these conventions consistently for high-quality, maintainable code.

## Core Principles

### Critical Requirements (Mandatory)
- **Always** use LLVM/MLIR naming conventions (detailed below)
- **Always** use `llvm::Error`/`llvm::Expected<T>` for error handling
- **Always** use LLVM casting (`dyn_cast`, `cast`, `isa`) instead of C-style casts
- **Always** handle errors and edge cases properly

### Preferred Patterns
- Write readable, maintainable code following the principle of least surprise
- Make code self-documenting through clear naming
- Use LLVM containers (`SmallVector`, `DenseMap`, `StringMap`) for performance
- Prefer `SmallVector<T, N>` for small collections
- Follow LLVM/MLIR C++ best practices

## Naming Conventions
```cpp
// ✅ Functions and variables: camelCase (lowercase start)
int calculateOffset();
StringRef fileName;
bool isValid;

// ✅ Types, classes, enums: CamelCase (uppercase start)
class MemoryBuffer;
enum TokenKind { Identifier, NumericConstant };
constexpr unsigned MaxBufferSize = 1024;

// ✅ Exception: STL-style methods use lowercase
class MyContainer {
  iterator begin();
  size_t size() const;
};

// ❌ Avoid: snake_case
int calculate_offset();  // Wrong
StringRef file_name;     // Wrong
```

## Code Structure and Organization
```cpp
// ✅ Include order: main header, local, LLVM, system
#include "MyFile.h"
#include "LocalHelper.h"
#include "llvm/IR/Function.h"
#include <algorithm>

// ✅ Function design: single responsibility, inputs first, return values
std::unique_ptr<Module> parseModule(StringRef name);

// ✅ Out-parameters acceptable when necessary
bool parseModule(StringRef name, Module &result);
```

**Key principles:**
- Use `#pragma once` or include guards
- Forward declare to reduce dependencies
- Make headers self-contained

## Memory Management
**Key principles:** Use LLVM types (`StringRef`, `ArrayRef`), express ownership clearly, follow RAII, use LLVM casting, avoid exceptions.

```cpp
// ✅ StringRef for read-only strings, ArrayRef for arrays
void processName(StringRef name);
void processData(ArrayRef<int> data);

// ✅ LLVM casting: dyn_cast for runtime checks, cast when type guaranteed
if (auto *call = dyn_cast<CallInst>(instr)) {
  // Handle call instruction
}
CallInst *call = cast<CallInst>(instr);  // Asserts if wrong type

// ❌ Avoid: C-style casts
CallInst *call = (CallInst*)instr;

// ✅ Smart pointers express ownership clearly
std::unique_ptr<Module> createModule();

// ❌ Avoid: Raw pointers for ownership
Module *createModule();  // Who owns this?

// ✅ References for non-null, pointers when null valid
void process(Module &mod);        // Non-null
void maybeProcess(Module *mod);   // Can be nullptr
```

## Error Handling
Use `llvm::Error` and `llvm::Expected<T>` for all error handling. Avoid exceptions.

```cpp
// ✅ Return Expected<T> for operations that can fail
Expected<std::unique_ptr<Module>> parseModule(StringRef filename);

// ✅ Check and propagate errors
auto moduleOrErr = parseModule(filename);
if (!moduleOrErr)
  return moduleOrErr.takeError();
```

## Performance Considerations
Avoid copies, use efficient algorithms, prefer LLVM containers, use range-based loops.

```cpp
// ✅ Pass by const reference, use LLVM containers
void processInstructions(const SmallVectorImpl<Instruction*> &instructions);

// ✅ Range-based loops over indexed loops
for (const BasicBlock &block : func) {
  for (const Instruction &inst : block) {
    // Process
  }
}

// ❌ Avoid: String copies, inefficient loops
void processName(std::string name);  // Use StringRef
for (int i = 0; i < vec.size(); ++i)  // Use range-based loop
```

## Documentation and Comments
Document public APIs and complex logic. Explain "why" not just "what".

```cpp
/// Analyzes the function for optimization opportunities.
/// Performs control flow analysis to identify safe optimizations.
SmallVector<Optimization> analyzeFunction(Function &func,
                                          const TargetTransformInfo &tti);
```

## Common Anti-Patterns to Avoid
```cpp
// ❌ Avoid: Hungarian notation
int m_count;  // Use: int count

// ❌ Avoid: Unnecessary complexity
if (condition == true)  // Use: if (condition)

// ❌ Avoid: Unclear auto usage
auto thing = process();  // What type is this?

// ✅ Good: Clear auto usage
auto *func = dyn_cast<Function>(val);
auto count = vec.size();  // Type is obvious
```

## Triton/MLIR-Specific Guidelines

```cpp
// ✅ Operation and pass naming: CamelCase with uppercase start
triton::LoadOp
triton::gpu::ConvertLayoutOp
class RemoveLayoutConversionsPass : public PassWrapper<...>
```

**GPU kernels:** Respect hardware limits (thread blocks, shared memory), use memory coalescing, avoid bank conflicts.

**MLIR patterns:** Use rewriter patterns for transformations, follow dialect conversion infrastructure, register and verify operations properly.
