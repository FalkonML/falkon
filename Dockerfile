FROM continuumio/miniconda3

# GCC
RUN add-apt-repository ppa:ubuntu-toolchain-r/test --yes
RUN apt-get -qqy update && apt-get -qqy install gcc-7 g++-7 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                        --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN update-alternatives --config gcc

# Base Packages
RUN apt-get -qqy update \
    && apt-get -qqy --no-install-recommends install \
    curl \
    wget \
    xvfb \
    git  \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Configure git
RUN git config --global user.email jenkins@example.com
RUN git config --global user.name jenkins-doc-updater

# Install pandoc
ARG PANDOC_VERSION=2.10
RUN mkdir -p /opt/pandoc \
    && wget -q "https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-linux-amd64.tar.gz" -O /opt/pandoc/pandoc.tar.gz \
    && tar xvzf /opt/pandoc/pandoc.tar.gz --strip-components 1 -C /opt/pandoc \
    && ln -fs /opt/pandoc/bin/pandoc /usr/bin/pandoc

# Conda Packages
#RUN conda install --yes --channel anaconda cmake

CMD ["/bin/bash"]
