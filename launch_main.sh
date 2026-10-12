#!/usr/bin/env bash
set -euo pipefail

# Run from the repository directory so relative imports and output paths remain
# consistent regardless of the shell's current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

timestamp=$(date +"%Y%m%d_%H%M%S")
python3 main.py \
  > "${timestamp}_stdout.log" \
  2> "${timestamp}_stderr.log"
