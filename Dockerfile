#FROM nvidia/cuda:10.2-devel-ubuntu18.04
FROM ubuntu:16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# CUDA drivers (taken from nvidia/cuda
RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl \
	&& NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 \
	&& NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 \
	&& apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub \
	&& apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub \
	&& echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - \
	&& rm cudasign.pub \
	&& echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
	&& echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list \
	&& apt-get purge --auto-remove -y gnupg-curl \
	&& rm -rf /var/lib/apt/lists/* # buildkit


ENV CUDA_VERSION=10.2.89
ENV CUDA_PKG_VERSION=10-2=10.2.89-1

# CUDA runtime
RUN apt-get update \
	&& apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-2 \
	&& ln -s cuda-10.2 /usr/local/cuda \
	&& rm -rf /var/lib/apt/lists/* # buildkit

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
	&& echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA=cuda>=10.2 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441
ENV NCCL_VERSION=2.9.6

RUN apt-get update \
	&& apt-get install -y --no-install-recommends cuda-libraries-$CUDA_PKG_VERSION cuda-npp-$CUDA_PKG_VERSION cuda-nvtx-$CUDA_PKG_VERSION libcublas10=10.2.2.89-1 libnccl2=$NCCL_VERSION-1+cuda10.2 \
	&& apt-mark hold libnccl2 \
	&& rm -rf /var/lib/apt/lists/*

RUN apt-get update \
	&& apt-get install -y --no-install-recommends cuda-nvml-dev-$CUDA_PKG_VERSION cuda-command-line-tools-$CUDA_PKG_VERSION cuda-nvprof-$CUDA_PKG_VERSION cuda-npp-dev-$CUDA_PKG_VERSION cuda-libraries-dev-$CUDA_PKG_VERSION \cuda-minimal-build-$CUDA_PKG_VERSION libcublas-dev=10.2.2.89-1 libnccl-dev=2.9.6-1+cuda10.2 \
	&& apt-mark hold libnccl-dev \
	&& rm -rf /var/lib/apt/lists/* # buildkit

# Base Packages
RUN apt-get -qqy update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
RUN apt-get -qqy update \
    && apt-get -qqy --no-install-recommends install \
    gcc-7 \
    g++-7 \
    curl \
    wget \
    xvfb \
    git  \
    libxml2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
# Configure gcc 7
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 60

RUN update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
RUN update-alternatives --set c++ /usr/bin/g++

RUN update-alternatives --config gcc
RUN update-alternatives --config g++

# Configure git
RUN git config --global user.email jenkins@example.com
RUN git config --global user.name jenkins-doc-updater

# Install pandoc
ARG PANDOC_VERSION=2.10
RUN mkdir -p /opt/pandoc \
    && wget -q "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz" -O /opt/pandoc/pandoc.tar.gz \
    && tar xvzf /opt/pandoc/pandoc.tar.gz --strip-components 1 -C /opt/pandoc \
    && ln -fs /opt/pandoc/bin/pandoc /usr/bin/pandoc

# Install various CUDA versions
RUN wget --quiet https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux \
    && chmod +x cuda_9.2.148_396.37_linux \
    && bash cuda_9.2.148_396.37_linux --silent --toolkit --no-opengl-libs --no-man-page --no-drm --toolkitpath="/opt/cuda9.2" \
    && rm cuda_9.2.148_396.37_linux

RUN wget --quiet https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run \
    && chmod +x cuda_10.1.243_418.87.00_linux.run \
    && bash cuda_10.1.243_418.87.00_linux.run --silent --toolkit --no-opengl-libs --no-man-page --no-drm --toolkitpath="/opt/cuda10.1" \
    && rm cuda_10.1.243_418.87.00_linux.run

RUN wget --quiet https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run \
    && chmod +x cuda_10.2.89_440.33.01_linux.run \
    && bash cuda_10.2.89_440.33.01_linux.run --silent --toolkit --no-opengl-libs --no-man-page --no-drm --toolkitpath="/opt/cuda10.2" \
    && rm cuda_10.2.89_440.33.01_linux.run

#RUN wget --quiet https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run \
#    && chmod +x cuda_11.0.3_450.51.06_linux.run \
#    && bash cuda_11.0.3_450.51.06_linux.run --silent --toolkit --no-opengl-libs --no-man-page --no-drm --toolkitpath="/opt/cuda11.0" \
#    && rm cuda_11.0.3_450.51.06_linux.run

#RUN wget --quiet https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run \
#    && chmod +x cuda_11.1.1_455.32.00_linux.run \
#    && bash cuda_11.1.1_455.32.00_linux.run --silent --toolkit --no-opengl-libs --no-man-page --no-drm --toolkitpath="/opt/cuda11.1" \
#    && rm cuda_11.1.1_455.32.00_linux.run


# Install miniconda3
RUN apt-get update -q && \
    apt-get install -q -y \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean
ENV PATH /opt/conda/bin:$PATH
ARG CONDA_VERSION=py38_4.9.2
ARG CONDA_MD5=122c8c9beb51e124ab32a0fa6426c656
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_MD5}  miniconda.sh" > miniconda.md5 && \
    if ! md5sum --status -c miniconda.md5; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Install conda packages
RUN conda install --yes --channel anaconda cmake

CMD ["/bin/bash"]

