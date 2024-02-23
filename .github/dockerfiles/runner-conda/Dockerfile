ARG INSTALLER_IMAGE=docker-registry.docker-registry.svc.cluster.local:5000/runner-conda:2024.0.1

FROM $INSTALLER_IMAGE as installer

FROM summerwind/actions-runner:ubuntu-22.04

USER root

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
    curl -s https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-opencl-icd \
      clinfo \
      intel-level-zero-gpu \
      level-zero \
      level-zero-dev libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev \
      build-essential \
      zlib1g-dev \
      cmake \
      ninja-build \
      ncurses-term \
      wget \
      pkg-config \
      libpng-dev libjpeg-dev libsndfile1-dev libxml2-dev libxslt1-dev \
      libgl1-mesa-glx  \
      fontconfig libfontconfig1-dev \
      libpango-1.0-0 libpangoft2-1.0-0 \
      libsdl2-dev libsdl2-2.0-0 \
    ; \
    rm -rf /var/lib/apt/lists/*

USER runner
SHELL ["/bin/bash", "-xec"]
ENV BASE=$HOME

COPY --from=installer /l_BaseKit*.sh $HOME/

# TODO: install only necessary components

WORKDIR $HOME

RUN \
  /bin/sh l_BaseKit*.sh -a --silent --eula accept; \
  rm l_BaseKit*.sh

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && sh Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="$PATH:$HOME/miniconda3/bin"

