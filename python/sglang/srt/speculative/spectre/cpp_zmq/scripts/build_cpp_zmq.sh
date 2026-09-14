#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"
INCLUDE_DIR="${ZMQ_INCLUDE_DIR:-/usr/include}"
"$PYTHON" -c 'import pybind11; import setuptools'
if [[ ! -f "$INCLUDE_DIR/zmq.hpp" ]]; then
  echo "Missing cppzmq header: $INCLUDE_DIR/zmq.hpp. Install cppzmq-dev or set ZMQ_INCLUDE_DIR." >&2
  exit 1
fi
cd "$PROJECT_DIR"
"$PYTHON" setup.py build_ext --inplace --force
