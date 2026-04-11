#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

/opt/anaconda3/bin/conda run -n deepsteg python xunet.py --opt options/steganalysis/train.yaml
