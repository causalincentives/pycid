#!/bin/bash

set -e

cd "$(git rev-parse --show-toplevel)"   # go to root directory

python3 -m mypy .
echo "passed type test"

python3 -m flake8
echo "passed lint"

python3 -m pytest
echo "passed unit test"

exit 0
