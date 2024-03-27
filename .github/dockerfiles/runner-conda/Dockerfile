ARG INSTALLER_IMAGE=docker-registry.docker-registry.svc.cluster.local:5000/runner-conda:2024.0.1

FROM $INSTALLER_IMAGE as installer
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN set -ex; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      gpg
      curl \
      jq

RUN set -ex; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified' > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
    curl -s https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main' > /etc/apt/sources.list.d/github-cli.list; \
    curl -sSL https://cli.github.com/packages/githubcli-archive-keyring.gpg >/usr/share/keyrings/githubcli-archive-keyring.gpg; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends --fix-missing \
      intel-opencl-icd \
      clinfo \
      intel-level-zero-gpu \
      level-zero \
      level-zero-dev libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev \
      gh \
    ; \
    rm -rf /var/lib/apt/lists/*

USER runner
SHELL ["/bin/bash", "-xec"]

COPY --from=installer /l_BaseKit*.sh $HOME/

# TODO: install only necessary components

WORKDIR $HOME

RUN \
  /bin/sh l_BaseKit*.sh -a --silent --eula accept; \
  rm l_BaseKit*.sh

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && sh Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="$PATH:$HOME/miniconda3/bin"
