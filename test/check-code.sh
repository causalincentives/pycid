#!/bin/bash

set -e

cd `git rev-parse --show-toplevel`   # go to root directory

python3 -m mypy . --config-file=test/lint-config.ini
echo "passed type test"

python3 -m flake8 --config=test/lint-config.ini .
echo "passed lint"

python3 -m unittest
echo "passed unit test"

exit 0