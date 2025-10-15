# Intel® XPU Backend for Triton\* on Windows

Intel® XPU Backend for Triton\* support for Windows is currently experimental and requires compiling PyTorch and Intel® XPU Backend for Triton\* from source.

## 1. Prerequisites

* Windows 11

* GPU card
  * Intel® Arc™ A-Series Graphics for Desktops
    * [Intel® Arc™ A750](https://www.intel.com/content/www/us/en/products/sku/227954/intel-arc-a750-graphics/specifications.html)
    * [Intel® Arc™ A770](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
  * Intel® Arc™ B-Series Graphics for Desktops
    * [Intel® Arc™ B570](https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html)
    * [Intel® Arc™ B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)

* Latest [GPU driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

In the following sections, all commands need to be executed in [PowerShell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.5).

## 2. Tools and dependencies

### Enable Windows long paths

Enable long paths for Windows file system.

```
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1
```

Reboot your computer to apply the change to the file system.

### Microsoft Visual Studio 2022

Install Microsoft Visual Studio 2022 and make sure the following [components](https://learn.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-community?view=vs-2022&preserve-view=true) are installed:
  * Microsoft.VisualStudio.Component.VC.Tools.x86.x64
  * Microsoft.VisualStudio.Component.Windows11SDK.22621
  * Microsoft.VisualStudio.Component.VC.CMake.Project

### Intel® Deep Learning Essentials

Install [Intel® Deep Learning Essentials 2025.2.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-essentials-os=windows&dl-win=offline).
By default, it is installed to `C:\Program Files (x86)\Intel\oneAPI`.

### Chocolatey

```
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Restart your PowerShell session to make sure `chocolatey` is properly initialized.

### Python

If you do not have a system Python installed at this step, you can install one with `chocolatey`.
For example:

```
choco install python --version=3.10.11
```

### Git

Install `git` with other POSIX tools, such as `bash`:

```
choco install -y git.install --params "'/GitAndUnixToolsOnPath /WindowsTerminal /NoAutoCrlf'"
```

Restart your PowerShell session to make sure `git` and other tools are properly initialized.

Enable symbolic links and long paths:

```
git config --global core.longpaths true
git config --global core.symlinks true
```

### Ninja

```
choco install -y ninja
```

### Pscx

The module [Pscx](https://github.com/Pscx/Pscx) is required for `Invoke-BatchFile`, a command to call bat/cmd script and set environment variables in the existing PowerShell session.

```
Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force
Install-Module Pscx -Scope CurrentUser -Force -AllowClobber
```

## 3. Build environment

Clone repository:

```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
```

Create a new virtual environment:

```
cd intel-xpu-backend-for-triton
python -m venv .venv
```


## 4. PyTorch

Activate the virtual environment:

```
.venv\Scripts\activate.ps1
```

Initialize Intel® Deep Learning Essentials, for example:

```
Invoke-BatchFile "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Build and install PyTorch:

```
$env:CMAKE_SHARED_LINKER_FLAGS = "/FORCE:MULTIPLE"
$env:CMAKE_MODULE_LINKER_FLAGS = "/FORCE:MULTIPLE"
$env:CMAKE_EXE_LINKER_FLAGS = "/FORCE:MULTIPLE"
bash -c "./scripts/install-pytorch.sh --source"
```

Check that PyTorch is installed:

```
python -c 'import torch;print(torch.__version__)'
```

## 5. Triton

Install build dependencies:

```
pip install -U wheel pybind11 cmake
```

Build and install Triton:

```
pip install -v '.[build,tests,tutorials]'
cd ..
```

Check that Triton is installed:

```
python -c 'import triton; print(triton.__version__)'
```

## 6. New PowerShell session

In a new PowerShell session, make sure the current directory is `intel-xpu-backend-for-triton` (a clone of the repository).
Initialize environment variables:

```
.venv\Scripts\activate.ps1
Invoke-BatchFile "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Check that PyTorch and Triton are available:

```
python -c 'import torch;print(torch.__version__)'
python -c 'import triton; print(triton.__version__)'
```
