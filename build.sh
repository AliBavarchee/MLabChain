#!/usr/bin/env bash
set -euo pipefail
cmake -S cpp -B build
cmake --build build --config Release
