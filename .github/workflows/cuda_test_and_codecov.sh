#!/bin/bash

set +euxo

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 3

pytest --cov-report=term-missing --cov-report=xml:coverage.xml --cov=falkon --cov-config setup.cfg

echo "$(date) || Uploading test-data to codecov..."
curl -s https://codecov.io/bash | bash -s -- -c -f coverage.xml -t "$CODECOV_TOKEN"
echo "$(date) || Data uploaded."
