# Targeted use cases

## 1. Functional regression testing for Dev IGC, NEO and Agama / Koubuk (Rolling) builds
## 2. Functional issues bisection by GTA Validation Team
## 3. Functional issues bisection by GTA Validation Team / IGC Team
## 4. Triton Env creation IGC Team

# Requirements

## 1. Self contained one file installer which should include everything needed to run Triton UT and Tutorials which should include:

Assumptions on the runtime:

- Bare metal machine
- Vanilla OS installation - Ubuntu 22.04 only in the initial implementation
- GPU HW - PVC / BMG
- GPU drivers are installed (UMD and KMD)

### 1.1 Python, to be independent of the system Python version

- Use minimal supported Python version. 3.10 as of now

### 1.2 Triton wheel for the selected Python version

- Use wheel from the latest nightly set

### 1.3 Triton test / scripts set which matches Triton wheel by commit id

- Implementation approach: include repo snapshot by commit id

### 1.4 Matching nightly pytorch wheel

- Use wheel from the latest nightly set

### 1.5 Matching oneAPI package (DLE)

- Publicly available latest DDLE and PTI versions should be used, use of the pre-release bits is out of scope for the initial implementation

### 1.6 All PyPI wheels which are needed by Triton / PyTorch runtimes plus test dependencies

- Implementation approach: create full env, remove triton and pytorch packages, run pip freeze and pip download

### 1.7 ABN metadata for GTA

- [Old approach] Generate ABN metadata based on installer test results
- [New approach] Generate ABN metadata based on the test reports for nightly wheels by commit id

## 2. GTA technical integration

### 2.1 Documentation / Links

- https://wiki.ith.intel.com/display/VPGOpenCL/Creating+GTA+TI+using+DynamicTest
- https://wiki.ith.intel.com/display/VPGOpenCL/Requests+to+Compute+PostSI+Validation
- Initial integration request: https://jira.devtools.intel.com/browse/VCV-12520

### 2.2 GTA asset upload

Assets to upload:

- self-contained installer
- run_triton_tests.sh
- pass_rate_utils.py
- abn_metadata.json

Command to upload:

```
gta-asset push gfx-ocl-abn-assets-igk/oneAPI/linux triton-xpu 2024100801 triton-xpu-installer --user vpenkin --root-url=https://gfx-assets.igk.intel.com/artifactory
```
