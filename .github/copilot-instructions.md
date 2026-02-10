# Code Review Copilot Custom Instructions

## Overview
This document provides custom instructions for AI-assisted code reviews following LLVM coding guidelines and best practices. Use these guidelines to ensure consistent, high-quality code reviews that align with LLVM project standards.

## Core Principles

### 1. Code Quality Focus
- Prioritize correctness, readability, and maintainability
- Ensure code follows the principle of least surprise
- Verify that code is self-documenting where possible
- Check for proper error handling and edge cases

### 2. LLVM Coding Standards Compliance
- Verify proper use of LLVM and MLIR data structures and APIs
- Use LLVM containers (SmallVector, DenseMap, StringMap) for better performance
- Prefer SmallVector<T, N> for small, frequently-used collections
- Ensure compliance with C++ best practices as adopted by LLVM/MLIR

## Naming Conventions

### Variables and Functions
```cpp
// ✅ Correct: CamelCase starting with a lowercase letter for functions and variables
int calculateOffset();
StringRef fileName;
bool isValid;

// ❌ Incorrect: snake_case or other conventions
int calculate_offset();
StringRef file_name;
bool is_valid;
```

### Types and Classes
```cpp
// ✅ Correct: CamelCase starting with uppercase
class MemoryBuffer;
struct SourceLocation;
enum TokenKind;

// ❌ Incorrect: lowercase or snake_case
class memory_buffer;
struct sourcelocation;
```

### Special Cases
```cpp
// ✅ Correct: Standard container-like methods use lowercase
class MyContainer {
  iterator begin();
  iterator end();
  size_t size() const;
};
```

### Constants and Enumerators
```cpp
// ✅ Correct: CamelCase starting with an uppercase letter for enum values
enum TokenKind {
  Identifier,
  NumericConstant,
  StringLiteral
};

// ✅ Correct: CamelCase starting with uppercase for static constants
static constexpr unsigned DefaultAlignment = 8;

// ✅ Also correct for constexpr constants
constexpr unsigned MaxBufferSize = 1024;
```

## Code Structure and Organization
### Header Files
- Check for proper include guards or #pragma once
- Verify forward declarations are used when possible
- Ensure headers are self-contained
- Validate proper ordering of includes:
  1. Main module header (for .cpp files)
  2. Local/private headers
  3. LLVM headers
  4. System headers

```cpp
// ✅ Correct include order
#include "MyFile.h"             // Main header
#include "LocalHelper.h"        // Local headers
#include "llvm/IR/Function.h"   // LLVM headers
#include <algorithm>            // System headers
```

### Function Design
- Verify functions have single, clear responsibilities
- Check parameter ordering: inputs first, then outputs
- Suggest proper use of const-correctness
- Validate return types (prefer returning values over out-parameters)

```cpp
// ✅ Preferred: Return by value
std::unique_ptr<Module> parseModule(StringRef name);

// ✅ Acceptable: out-parameter when necessary
bool parseModule(StringRef name, Module &result);
```

## Memory Management
### LLVM-Specific Patterns
- Verify proper use of StringRef instead of std::string for read-only strings
- Check for appropriate use of ArrayRef and MutableArrayRef
- Ensure proper ownership semantics with smart pointers
- Validate use of LLVM's casting system (dyn_cast, cast, isa)

```cpp
// ✅ Correct: Use StringRef for parameters
void processName(StringRef name);

// ✅ Correct: Use LLVM casting
if (auto *call = dyn_cast<CallInst>(instr)) {
  // Handle call instruction
}

// ❌ Incorrect: C-style cast
CallInst *call = (CallInst*)instr;
```

### Resource Management
- Verify RAII principles are followed
- Check for proper, deterministic cleanup in destructors and other RAII-managed objects
- Avoid relying on C++ exceptions; prefer RAII plus LLVM-style error propagation (`llvm::Error`/`llvm::Expected`)
- Validate proper use of move semantics

## Error Handling
### LLVM Error Handling Patterns
- Check for proper use of llvm::Error and llvm::Expected<T>
- Verify error propagation is handled correctly
- Ensure fatal errors use appropriate LLVM mechanisms

```cpp
// ✅ Correct: Using Expected<T>
Expected<std::unique_ptr<Module>> parseModule(StringRef filename);

// ✅ Correct: Error handling
auto moduleOrErr = parseModule(filename);
if (!moduleOrErr)
  return moduleOrErr.takeError();
```

## Performance Considerations
### Efficiency Guidelines
- Check for unnecessary copies (prefer move semantics)
- Verify appropriate use of references vs. pointers
- Ensure efficient container usage
- Validate algorithmic complexity (prefer O(n) over O(n²) where possible)
- Check for redundant operations in hot paths
- Verify loop invariant code motion opportunities

```cpp
// ✅ Efficient: Pass by const reference
void processInstructions(const std::vector<Instruction*> &instructions);

// ✅ Efficient: Use range-based loops
for (const BasicBlock &block : func) {
  for (const Instruction &inst : block) {
    // Process instruction
  }
}
```

## Documentation and Comments
### Comment Guidelines
- Verify complex algorithms are well-documented
- Check that public APIs have appropriate documentation
- Ensure comments explain "why" not just "what"
- Validate that comments are up-to-date with code changes

```cpp
/// Analyzes the given function for optimization opportunities.
///
/// This performs comprehensive control flow analysis and identifies
/// optimizations that can be applied safely.
SmallVector<Optimization> analyzeFunction(Function &func,
                                          const TargetTransformInfo &tti);
```

## Testing and Validation
### Test Coverage
- Verify that new functionality includes appropriate tests
- Check for edge case coverage
- Ensure tests follow LLVM testing conventions
- Validate that tests are deterministic and reliable

### Code Review Checklist
 - [ ] Follows MLIR/LLVM naming conventions
 - [ ] Appropriate error handling
 - [ ] Memory management follows MLIR/LLVM patterns
 - [ ] Proper include organization and dependencies
 - [ ] No obvious security vulnerabilities
 - [ ] Performance considerations addressed
 - [ ] Code is readable and maintainable
 - [ ] Documentation is clear and complete
 - [ ] Adequate test coverage

## Common Anti-Patterns to Flag
### Style Issues
```cpp
// ❌ Avoid: Hungarian notation or prefixes
int m_count;
bool bIsValid;

// ✅ Correct: CamelCase with lowercase start for members
class MyClass {
  int count;
  bool isValid;
  
  // Or use descriptive names that don't conflict
  int elementCount;
  bool valid;
};

// ❌ Avoid: Unnecessary complexity
if (condition == true)  // Should be if (condition)
```

### Modern C++ Features
```cpp
// ✅ Good use of auto with clear type
auto *func = dyn_cast<Function>(val);
auto count = vec.size();  // Type is obvious

// ❌ Avoid: Unclear types
auto thing = process();  // What type is Thing?

// ✅ Structured bindings for clarity
for (auto [key, value] : mapping) {
  // Use key and value
}
```

## Triton/MLIR-Specific Guidelines

### Operation and Pass Naming
```cpp
// ✅ Correct: Triton operations use CamelCase starting with an uppercase letter 
triton::LoadOp
triton::gpu::ConvertLayoutOp

// ✅ Correct: Passes use descriptive names
class RemoveLayoutConversionsPass : public PassWrapper<...>
```

### GPU Kernel Considerations
- Verify thread block size calculations are within hardware limits
- Check for proper memory coalescing patterns
- Validate shared memory usage and bank conflict avoidance
- Ensure proper synchronization primitives

### MLIR Patterns
- Use proper MLIR rewriter patterns for transformations
- Verify dialect conversions follow MLIR infrastructure
- Check that operations are properly registered and verified

### Performance Issues
```cpp
// ❌ Avoid: Unnecessary string copies
void func(std::string name);  // Should use StringRef

// ❌ Avoid: Inefficient loops
for (int i = 0; i < vec.size(); ++i)  // Prefer range-based loops
```

### Safety Issues
```cpp
// ❌ Avoid: Raw pointer ownership
Instruction *createInst();  // Should return unique_ptr or similar

// ❌ Avoid: Unchecked casts
CallInst *call = cast<CallInst>(instr);  // Use dyn_cast with null check when instr may not be a CallInst; use cast only when invariants guarantee it
```

## Review Response Guidelines
### When providing feedback:
- **Be Constructive:** Explain the reasoning behind suggestions
- **Provide Examples:** Show correct implementations when possible
- **Prioritize Issues:** Distinguish between critical issues and style preferences
- **Reference Standards:** Cite specific LLVM guidelines when applicable
- **Suggest Improvements:** Don't just identify problems, propose solutions

## Conclusion
These guidelines ensure that code reviews maintain LLVM/MLIR's high standards for code quality, performance, and maintainability. 
Always consider the broader impact of changes on the codebase and the developer experience.
