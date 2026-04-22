#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate warpembot
exec python "$(dirname "$0")/run.py" "$@"
