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
- Enforce LLVM naming conventions and style guidelines
- Verify proper use of LLVM data structures and APIs
- Ensure compliance with C++ best practices as adopted by LLVM

## Naming Conventions

### Variables and Functions
```cpp
// ✅ Correct: CamelCase for functions and variables
int calculateOffset();
StringRef FileName;
bool IsValid;

// ❌ Incorrect: snake_case or other conventions
int calculate_offset();
StringRef file_name;
bool is_valid;
```

### Types and Classes
``` cpp
// ✅ Correct: CamelCase starting with uppercase
class MemoryBuffer;
struct SourceLocation;
enum TokenKind;

// ❌ Incorrect: lowercase or snake_case
class memory_buffer;
struct sourcelocation;
```

### Constants and Enumerators
```cpp
// ✅ Correct: CamelCase for enum values
enum TokenKind {
  Identifier,
  NumericConstant,
  StringLiteral
};

// ✅ Correct: Static constants
static constexpr unsigned DefaultAlignment = 8;
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
#include "llvm/Analysis/TargetTransformInfo.h"  // Main header
#include "LocalHelper.h"                        // Local headers
#include "llvm/IR/Function.h"                   // LLVM headers
#include <algorithm>                            // System headers
```

### Function Design
- Verify functions have single, clear responsibilities
- Check parameter ordering: inputs first, then outputs
- Ensure proper use of const-correctness
- Validate return types (prefer returning values over out-parameters)

```cpp
// ✅ Preferred: Return by value
std::unique_ptr<Module> parseModule(StringRef Filename);

// ✅ Acceptable: Out-parameter when necessary
bool parseModule(StringRef Filename, Module &Result);
```

## Memory Management
### LLVM-Specific Patterns
- Verify proper use of StringRef instead of std::string for read-only strings
- Check for appropriate use of ArrayRef and MutableArrayRef
- Ensure proper ownership semantics with smart pointers
- Validate use of LLVM's casting system (dyn_cast, cast, isa)

```cpp
// ✅ Correct: Use StringRef for parameters
void processName(StringRef Name);

// ✅ Correct: Use LLVM casting
if (auto *CI = dyn_cast<CallInst>(I)) {
  // Handle call instruction
}

// ❌ Incorrect: C-style cast
CallInst *CI = (CallInst*)I;
```

### Resource Management
- Verify RAII principles are followed
- Check for proper cleanup in destructors
- Ensure exception safety where applicable
- Validate proper use of move semantics

## Error Handling
### LLVM Error Handling Patterns
- Check for proper use of llvm::Error and llvm::Expected<T>
- Verify error propagation is handled correctly
- Ensure fatal errors use appropriate LLVM mechanisms

```cpp
// ✅ Correct: Using Expected<T>
Expected<std::unique_ptr<Module>> parseModule(StringRef Filename);

// ✅ Correct: Error handling
auto ModuleOrErr = parseModule(Filename);
if (!ModuleOrErr)
  return ModuleOrErr.takeError();
```

## Performance Considerations
### Efficiency Guidelines
- Check for unnecessary copies (prefer move semantics)
- Verify appropriate use of references vs. pointers
- Ensure efficient container usage
- Validate algorithmic complexity considerations

```cpp
// ✅ Efficient: Pass by const reference
void processInstructions(const std::vector<Instruction*> &Instructions);

// ✅ Efficient: Use range-based loops
for (const auto &BB : F) {
  for (const auto &I : BB) {
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
/// \brief Analyzes the given function for optimization opportunities.
///
/// This function performs a comprehensive analysis of the control flow
/// and identifies potential optimizations that can be applied safely.
///
/// \param F The function to analyze
/// \param TTI Target-specific information for cost modeling
/// \return A list of suggested optimizations
std::vector<Optimization> analyzeFunction(Function &F,
                                         const TargetTransformInfo &TTI);
```

## Testing and Validation
### Test Coverage
- Verify that new functionality includes appropriate tests
- Check for edge case coverage
- Ensure tests follow LLVM testing conventions
- Validate that tests are deterministic and reliable

### Code Review Checklist
 - [ ] Follows LLVM naming conventions
 - [ ] Proper include organization and dependencies
 - [ ] Appropriate error handling
 - [ ] Memory management follows LLVM patterns
 - [ ] Performance considerations addressed
 - [ ] Adequate test coverage
 - [ ] Documentation is clear and complete
 - [ ] No obvious security vulnerabilities
 - [ ] Code is readable and maintainable

## Common Anti-Patterns to Flag
### Style Issues
```cpp
// ❌ Avoid: Inconsistent naming
void ProcessFile();  // Should be processFile()
int m_count;         // Should be Count

// ❌ Avoid: Unnecessary complexity
if (condition == true)  // Should be if (condition)
```

### Performance Issues
```cpp
// ❌ Avoid: Unnecessary string copies
void func(std::string Name);  // Should use StringRef

// ❌ Avoid: Inefficient loops
for (int i = 0; i < vec.size(); ++i)  // Prefer range-based loops
```

### Safety Issues
```cpp
// ❌ Avoid: Raw pointer ownership
Instruction *createInst();  // Should return unique_ptr or similar

// ❌ Avoid: Unchecked casts
CallInst *CI = cast<CallInst>(I);  // Should use dyn_cast with null check
```

## Review Response Guidelines
### When providing feedback:
- Be Constructive: Explain the reasoning behind suggestions
- Provide Examples: Show correct implementations when possible
- Prioritize Issues: Distinguish between critical issues and style preferences
- Reference Standards: Cite specific LLVM guidelines when applicable
- Suggest Improvements: Don't just identify problems, propose solutions

## Conclusion
These guidelines ensure that code reviews maintain LLVM's high standards for code quality, performance, and maintainability. Always consider the broader impact of changes on the codebase and the developer experience.
