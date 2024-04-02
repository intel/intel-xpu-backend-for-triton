ARG INSTALLER_IMAGE=docker-registry.docker-registry.svc.cluster.local:5000/oneapi-basekit:2024.0.1

FROM $INSTALLER_IMAGE as installer

FROM summerwind/actions-runner:ubuntu-22.04

USER root

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
    curl -sSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list;\
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get clean; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-opencl-icd=23.35.27191.42-775~22.04 \
      clinfo \
      intel-level-zero-gpu=1.3.27191.42-775~22.04 \
      level-zero=1.14.0-744~22.04 \
      level-zero-dev=1.14.0-744~22.04 \
      intel-igc-cm=1.0.206-775~22.04 \
      libigfxcmrt-dev=23.4.0-775~22.04 \
      libigc1=1.0.15136.24-775~22.04 \
      libigc-dev=1.0.15136.24-775~22.04 \
      libigdfcl1=1.0.15136.24-775~22.04 \
      libigdfcl-dev=1.0.15136.24-775~22.04 \
      \
      build-essential \
      zlib1g-dev \
      cmake \
      ninja-build \
      ncurses-term \
      pkg-config \
      \
      libpng-dev libjpeg-dev libsndfile1-dev libxml2-dev libxslt1-dev \
      libgl1-mesa-glx  \
      fontconfig libfontconfig1-dev \
      libpango-1.0-0 libpangoft2-1.0-0 \
      libsdl2-dev libsdl2-2.0-0 \
      \
      gh \
    ; \
    rm -rf /var/lib/apt/lists/*

USER runner
WORKDIR $HOME

SHELL ["/bin/bash", "-xec"]

COPY --from=installer /l_BaseKit*.sh $HOME/

# TODO: install only necessary components
RUN \
  /bin/sh l_BaseKit*.sh -a --silent --eula accept; \
  rm l_BaseKit*.sh

RUN \
  curl -sSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
  sh Miniconda3-latest-Linux-x86_64.sh -b; \
  rm Miniconda3-latest-Linux-x86_64.sh
