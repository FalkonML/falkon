#!/bin/bash

set -ex

mkdir -p /opt/cmake
wget -q "https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.tar.gz" -O /opt/cmake/cmake.tar.gz
tar -xvzf /opt/cmake/cmake.tar.gz --strip-components 1 -C /opt/cmake
rm /opt/cmake/cmake.tar.gz
export PATH=/opt/cmake/bin:$PATH
# This is probably useless
ln -fs /opt/cmake/bin/cmake /usr/bin/cmake
