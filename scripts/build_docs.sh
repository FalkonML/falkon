#!/bin/bash

set -ex

sh 'python -m pip install --upgrade --progress-bar off ghp-import'
withCredentials([string(credentialsId: 'GIT_TOKEN', variable: 'GIT_TOKEN')]) {
    sh "/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset"
    sh 'git remote set-url origin https://Giodiro:${GIT_TOKEN}@github.com/FalkonML/falkon.git'
    sh '''
    cd ./doc
    make clean && make html && make install
    '''
}
