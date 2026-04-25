#!/usr/bin/env bash
# Clean environment variables that can make PyTorch load CUDA/cuDNN from another venv.
#
# Usage:
#   source ./clean_env.sh
#   ./clean_env.sh ./venv/bin/python ./run_demo.py --check-only

unset LD_LIBRARY_PATH
unset LD_PRELOAD
unset PYTHONPATH

# Keep Python isolated from user site packages that may point to another project.
export PYTHONNOUSERSITE=1

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ "$#" -eq 0 ]]; then
    echo "Environment cleaned. To keep it in your current shell, run: source ./clean_env.sh"
    exit 0
  fi

  exec "$@"
fi
