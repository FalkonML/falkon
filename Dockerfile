ARG CUDA_VERSION=10.2
ARG BASE_TARGET=cuda${CUDA_VERSION}
FROM nvidia/cuda:9.2-devel-centos7 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN yum install -y wget curl util-linux xz bzip2 git patch which unzip tar
RUN yum install -y yum-utils centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran devtoolset-7-binutils
# Configure git
RUN git config --global user.email jenkins@example.com
RUN git config --global user.name jenkins-doc-updater
# EPEL for cmake
#RUN wget http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
#    rpm -ivh epel-release-latest-7.noarch.rpm && \
#    rm -f epel-release-latest-7.noarch.rpm
# cmake
#RUN yum install -y cmake3 && \
#    ln -s /usr/bin/cmake3 /usr/bin/cmake
#ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
#ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH

RUN yum install -y autoconf aclocal automake make sudo
RUN rm -rf /usr/local/cuda-*


#FROM base as openssl
# Install openssl
#ADD ./common/install_openssl.sh install_openssl.sh
#RUN bash ./install_openssl.sh && rm install_openssl.sh

FROM base as cmake
# Install cmake 3.18 (needed by keops)
ADD scripts/install_cmake.sh install_cmake.sh
RUN bash install_cmake.sh && rm install_cmake.sh

FROM base as conda
# Install Anaconda
ADD scripts/install_conda.sh install_conda.sh
RUN bash install_conda.sh && rm install_conda.sh

FROM base as pandoc
# Install pandoc
ADD scripts/install_pandoc.sh install_pandoc.sh
RUN bash install_pandoc.sh && rm install_pandoc.sh

FROM base as ghrelease
#Install github-release
ADD scripts/install_github_release.sh install_github_release.sh
RUN bash install_github_release.sh && rm install_github_release.sh

# Install CUDA
FROM base as cuda
RUN rm -rf /usr/local/cuda-*
ADD scripts/install_cuda.sh install_cuda.sh

FROM cuda as cuda9.2
RUN bash install_cuda.sh 9.2
ENV DESIRED_CUDA=9.2

FROM cuda as cuda10.1
RUN bash install_cuda.sh 10.1
ENV DESIRED_CUDA=10.1

FROM cuda as cuda10.2
RUN bash install_cuda.sh 10.2
ENV DESIRED_CUDA=10.2

FROM cuda as cuda11.0
RUN bash install_cuda.sh 11.0
ENV DESIRED_CUDA=11.0

FROM cuda as cuda11.1
RUN bash install_cuda.sh 11.1
ENV DESIRED_CUDA=11.1

FROM cuda as cuda11.2
RUN bash install_cuda.sh 11.2
ENV DESIRED_CUDA=11.2

FROM cuda as cuda11.3
RUN bash install_cuda.sh 11.3
ENV DESIRED_CUDA=11.3


FROM base as all_cuda
COPY --from=cuda9.2   /usr/local/cuda-9.2  /usr/local/cuda-9.2
COPY --from=cuda10.1  /usr/local/cuda-10.1 /usr/local/cuda-10.1
COPY --from=cuda10.2  /usr/local/cuda-10.2 /usr/local/cuda-10.2
COPY --from=cuda11.0  /usr/local/cuda-11.0 /usr/local/cuda-11.0
COPY --from=cuda11.1  /usr/local/cuda-11.1 /usr/local/cuda-11.1
COPY --from=cuda11.2  /usr/local/cuda-11.2 /usr/local/cuda-11.2
COPY --from=cuda11.3  /usr/local/cuda-11.3 /usr/local/cuda-11.3

FROM ${BASE_TARGET} as final
#COPY --from=openssl            /opt/openssl           /opt/openssl
COPY --from=conda              /opt/conda             /opt/conda
COPY --from=pandoc             /opt/pandoc            /opt/pandoc
COPY --from=ghrelease          /opt/github_release    /opt/github_release
COPY --from=cmake              /opt/cmake             /opt/cmake
ENV  PATH=/opt/conda/bin:$PATH
ENV  PATH=/opt/cmake/bin:$PATH
ENV  PATH=/opt/pandoc/bin:$PATH
RUN rm -rf /usr/local/cuda
RUN chmod o+rw /usr/local
RUN touch /.condarc && \
    chmod o+rw /.condarc && \
    chmod -R o+rw /opt/conda
