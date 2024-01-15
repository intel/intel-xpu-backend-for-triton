# Compiling OpenAI/triton on Nvidia Machines

1. ```git clone https://github.com/openai/triton.git```
2. ```cd triton```
3. ```pip install ninja cmake wheel```

If python is not installed you will have to complete the rest of the steps in a virtual environment using anaconda. Steps can be found here: https://docs.anaconda.com/free/anaconda/install/linux/

4. ```pip install -e python```

## Building custom LLVM

Back out of the triton repository to clone and build a custom llvm

5. ```git clone https://github.com/llvm/llvm-project.git```

Check https://github.com/openai/triton/blob/main/cmake/llvm-hash.txt to see the latest llvm hash to use in the next step

6. ```git reset --hard <hash>```
7. ```mkdir build```
8. ```cd build```
9. ```cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm"```
10. ```ninja```

## Build Triton

11. ```export LLVM_BUILD_DIR=$HOME/llvm-project/build```
12. ```cd <triton install>```
13. ```LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR pip install -e python```

Optional:
Set ```TRITON_BUILD_WITH_CLANG_LLD=true``` as an environment variable to use clang and lld. lld in particular results in faster builds.
Set ```TRITON_BUILD_WITH_CCACHE=true``` to build with ccache.

## Running Tests

14. ```pip install scipy numpy torch pytest lit && pip install -e python```

If this step fails and complains about the python version such as:
```ERROR: Package 'networkx' requires a different Python: 3.8.10 not in '>=3.9'```
You will need to download anaconda and complete this step as well as the following steps in a virtual environment
ex: https://docs.anaconda.com/free/anaconda/install/linux/

15. ```python3 -m pytest python/test/unit```

This will take about 20 minutes

If you get the error: ```ImportError: /home/username/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found```
Run: ```conda install -c conda-forge libstdcxx-ng```, source: https://stackoverflow.com/questions/58424974/anaconda-importerror-usr-lib64-libstdc-so-6-version-glibcxx-3-4-21-not-fo

Current result of test:
```408 failed, 6405 passed, 2887 skipped, 86 warnings in 1397.08s (0:23:17) ```

Full test output can be found in [triton-test-output.txt](https://github.com/intel/intel-xpu-backend-for-triton/files/13942330/triton-test-output.txt)

Reference: https://github.com/openai/triton
