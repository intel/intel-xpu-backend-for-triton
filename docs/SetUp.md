git clone -b v2.0.1 https://github.com/pytorch/pytorch.git
conda install cmake ninja mkl mkl-include

conda install -y -c conda-forge libstdcxx-ng

cd pytorch
pip install -r requirements.txt
git submodule sync && git submodule update --init --recursive


export _GLIBCXX_USE_CXX11_ABI=1


export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py bdist_wheel
pip install dist/torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
# IPEX

 git clone -b xpu-master https://github.com/intel/intel-extension-for-pytorch.git
git submodule sync && git submodule update --init --recursive
 source ~/tongsu/env_triton.sh
 python setup.py bdist_wheel