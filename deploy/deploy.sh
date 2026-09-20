#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/ubuntu/Food-Demand-Forcasting}"
SERVICE="${SERVICE:-foodflow}"

cd "$REPO_DIR"
git pull --ff-only
. .venv/bin/activate
pip install -q -r requirements.txt
sudo systemctl restart "$SERVICE"
echo "deployed $(git rev-parse --short HEAD) at $(date -Is)"
