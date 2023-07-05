# LLVM Build bug
## 1. error while loading shared libraries: libtinfow.so.6

Error message would be like:
```Bash
/llvm/build/bin/mlir-tblgen: error while loading shared libraries: libtinfow.so.6: cannot open shared object file: No such file or directory
```
You could locate your `libtinfow.so.6` and preload it.

```Bash
# First find your desired .so, normally it is under your conda env
locate libtinfow.so.6
# Preload it
export LD_PRELOAD={the-path-to}/libtinfow.so
```
