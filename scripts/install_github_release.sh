#!/bin/bash

set -ex

wget -q https://github.com/github-release/github-release/releases/download/v0.10.0/linux-amd64-github-release.bz2
bzip2 -d --stdout linux-amd64-github-release.bz2 > github_release
chmod a+x github_release
mkdir /opt/github_release
mv github_release /opt/github_release/
export PATH=/opt/github_release:$PATH
# This is most likely pointless
ln -fs /opt/github_release/github_release /usr/bin/github_release

