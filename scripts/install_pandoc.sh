#!/bin/bash

set -ex

mkdir -p /opt/pandoc
wget -q "https://github.com/jgm/pandoc/releases/download/2.10/pandoc-2.10-linux-amd64.tar.gz" -O /opt/pandoc/pandoc.tar.gz
tar -xvzf /opt/pandoc/pandoc.tar.gz --strip-components 1 -C /opt/pandoc
rm /opt/pandoc/pandoc.tar.gz
export PATH=/opt/pandoc/bin:$PATH
# This is probably useless
ln -fs /opt/pandoc/bin/pandoc /usr/bin/pandoc

