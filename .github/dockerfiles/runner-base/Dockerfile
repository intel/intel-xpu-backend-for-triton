FROM summerwind/actions-runner:ubuntu-22.04

USER root

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
    curl -sSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list;\
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-opencl-icd \
      clinfo \
      intel-opencl-icd intel-level-zero-gpu level-zero \
      intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 libigdgmm12 \
      libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev \
      \
      build-essential \
      zlib1g-dev \
      cmake \
      ninja-build \
      ncurses-term \
      pkg-config \
      wget \
      \
      libpng-dev libjpeg-dev libsndfile1-dev libxml2-dev libxslt1-dev \
      fontconfig libfontconfig1-dev \
      libpango-1.0-0 libpangoft2-1.0-0 \
      \
      gh \
    ; \
    apt-get install -y --no-install-recommends --fix-missing \
      libgl1 \
    ; \
    rm -rf /var/lib/apt/lists/*

USER runner
WORKDIR $HOME

SHELL ["/bin/bash", "-xec"]

RUN \
   mkdir -p ~/.local/bin; \
   curl -sSL https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz | tar Jxf - -C ~/.local/bin --strip-components 3 --no-anchored spirv-dis; \
   spirv-dis --version

RUN \
  curl -sSLO https://registrationcenter-download.intel.com/akdlm/IRC_NAS/6758ce21-bde6-47f8-b4bf-6df4301f187d/l_intel-for-pytorch-gpu-dev_p_0.5.1.40_offline.sh; \
  sh l_intel-for-pytorch-gpu*.sh -a --silent --eula accept; \
  rm l_intel-for-pytorch-gpu*.sh

RUN \
  curl -sSLO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh; \
  sh Miniforge3-Linux-x86_64.sh -b; \
  rm Miniforge3-Linux-x86_64.sh
