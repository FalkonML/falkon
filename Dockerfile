FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

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

# Conda Packages
RUN conda install --yes --channel anaconda cmake

# GHP-Import (for docs)
RUN python -m pip install --upgrade --progress-bar off ghp-import;

# Install pandoc
ARG PANDOC_VERSION=2.10
RUN mkdir -p /opt/pandoc \
    && wget -q "https://github.com/jgm/pandoc/releases/download/$PANDOC_VERSION/pandoc-$PANDOC_VERSION-linux-amd64.tar.gz" -O /opt/pandoc/pandoc.tar.gz \
    && tar xvzf /opt/pandoc/pandoc.tar.gz --strip-components 1 -C /opt/pandoc \
    && ln -fs /opt/pandoc/bin/pandoc /usr/bin/pandoc

CMD ["/bin/bash"]