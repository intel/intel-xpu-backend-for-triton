ARG INSTALLER_IMAGE=docker-registry.docker-registry.svc.cluster.local:5000/oneapi-basekit:2023.2.0

FROM $INSTALLER_IMAGE as installer

FROM summerwind/actions-runner:ubuntu-22.04

USER root

RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy max' > /etc/apt/sources.list.d/intel-graphics.list; \
    curl -s https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-opencl-icd \
      clinfo \
    ; \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/intel-oneapi.list; \
    curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-level-zero-gpu \
      level-zero \
      level-zero-dev \
    ; \
    apt-get install -y --no-install-recommends --fix-missing \
      build-essential \
      zlib1g-dev \
      cmake \
      ninja-build \
      ncurses-term \
      wget \
    ; \
    rm -rf /var/lib/apt/lists/*

USER runner
SHELL ["/bin/bash", "-xec"]
ENV BASE=$HOME

COPY --from=installer /l_BaseKit*.sh $HOME/

# TODO: install only necessary components

RUN \
  cd $HOME; \
  /bin/sh l_BaseKit*.sh -a --silent --eula accept; \
  rm l_BaseKit*.sh; \
  echo "source $HOME/intel/oneapi/setvars.sh --force > /dev/null" > $HOME/.bashrc

