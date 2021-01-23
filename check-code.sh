#!/bin/bash

set -e

python3 -m mypy .
echo "passed type test"

python3 -m flake8 .
echo "passed lint"

python3 -m unittest
echo "passed unit test"

exit 0