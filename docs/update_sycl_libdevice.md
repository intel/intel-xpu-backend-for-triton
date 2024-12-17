# Guide to Update SYCL Device Library

This guide will walk you through the steps to update the SYCL device library using the Intel DPC++ compiler.

## Step 1: Display Commands used during Compilation Process
1. Open a terminal.
2. Run the following command to compile a C++ file:
```sh
dpcpp -save-temps -#x t.cpp
```
Replace t.cpp with any C++ file of your choice. This command will display the commands used during the compilation process.

## Step 2: Locate the llvm-link Command
From the output of the previous command, find the llvm-link command line. It should look similar to the following example:
```sh
"/opt/intel/oneapi/compiler/2025.0/bin/compiler/llvm-link" \
    -only-needed \
    t-sycl-spir64-unknown-unknown-b331ea.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-crt.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-complex.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-complex-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-cmath.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-cmath-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf-bf16.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cassert.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cstring.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-complex.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-complex-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cmath.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cmath-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf-bf16.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-user-wrappers.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-compiler-wrappers.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-stubs.bc \
    -o \
    t-sycl-spir64-unknown-unknown-d81f68.bc \
    --suppress-warnings
```

## Step 3: Modify the llvm-link Command
Remove the `-only-needed` option and the intermediate file `t-sycl-spir64-unknown-unknown-b331ea.bc` from the command line.
And modify to output file name to `libsycl-spir64-unknown-unknown.bc`.
The modified command should look like this:
```sh
"/opt/intel/oneapi/compiler/2025.0/bin/compiler/llvm-link" \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-crt.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-complex.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-complex-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-cmath.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-cmath-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-imf-bf16.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cassert.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cstring.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-complex.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-complex-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cmath.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-cmath-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf-fp64.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-fallback-imf-bf16.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-user-wrappers.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-compiler-wrappers.bc \
    /opt/intel/oneapi/compiler/2025.0/bin/compiler/../../lib/libsycl-itt-stubs.bc \
    -o \
    libsycl-spir64-unknown-unknown.bc \
    --suppress-warnings
```

## Step 4: Execute the Modified Command
Copy the modified llvm-link command.
Paste and run it in the terminal.

## Step 5: Check for Manual Changes
Check the log of the existing device library to see what manual changes need to be made:
```sh
git log third_party/intel/backend/lib/libsycl-spir64-unknown-unknown.bc
```
Look for any specific changes mentioned in the commit messages. For example, from commit 0dd37fc92c46f35c6ced34801e51058b6b89ea47, you need to change one of the module metadata from 4 to 3.

## Step 6: Apply Manual Changes
`llvm-dis` to disassemble the bitcode library, then based on the information from the git log, apply the necessary manual changes to the updated device library.
Reassemble the modified LLVMIR device library using `llvm-as`.

By following these steps, you will have successfully updated the SYCL device library and applied any necessary manual changes.
