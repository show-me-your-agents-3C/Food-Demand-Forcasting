#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:show-me-your-agents-3C/Food-Demand-Forcasting.git}"
DOMAIN="${DOMAIN:-}"
REPO_DIR="${REPO_DIR:-$HOME/Food-Demand-Forcasting}"
SERVICE="${SERVICE:-foodflow}"
PYTHON="${PYTHON:-python3}"

if [ ! -d "$REPO_DIR/.git" ]; then
  export GIT_LFS_SKIP_SMUDGE=1
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" config lfs.fetchexclude '*'

cd "$REPO_DIR"

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in the gateway credentials."
fi

sed "s#/home/ubuntu/Food-Demand-Forcasting#$REPO_DIR#g" deploy/foodflow.service | sudo tee "/etc/systemd/system/${SERVICE}.service" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE"

if [ -n "$DOMAIN" ]; then
  sed "s#YOUR_DOMAIN.example.com#$DOMAIN#g; s#/home/ubuntu/Food-Demand-Forcasting#$REPO_DIR#g" deploy/Caddyfile | sudo tee /etc/caddy/Caddyfile >/dev/null
  sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy
fi

echo "bootstrap complete"
echo "check: curl -s http://127.0.0.1:8000/health"
