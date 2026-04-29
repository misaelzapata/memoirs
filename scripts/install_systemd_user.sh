#!/usr/bin/env bash
# Install user-level systemd units so memoirs auto-starts on login (no
# root needed). Two units:
#   memoirs-api.service    — HTTP API + Web UI on port 8283
#   memoirs-mcp@.socket    — on-demand MCP stdio server (only spawned when
#                            an IDE actually connects)
#
# Usage:
#   bash scripts/install_systemd_user.sh        # install + enable + start
#   bash scripts/install_systemd_user.sh --uninstall
#
# After install:
#   systemctl --user status memoirs-api
#   journalctl --user -u memoirs-api -f
#   systemctl --user disable --now memoirs-api    # to stop auto-start

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MEMOIRS_VENV:-$ROOT/.venv}"
DB="${MEMOIRS_DB:-$ROOT/.memoirs/memoirs.sqlite}"
PORT="${MEMOIRS_PORT:-8283}"
UNIT_DIR="${HOME}/.config/systemd/user"

if [ "${1:-}" = "--uninstall" ]; then
    systemctl --user disable --now memoirs-api 2>/dev/null || true
    rm -f "$UNIT_DIR/memoirs-api.service"
    systemctl --user daemon-reload
    echo "✓ uninstalled"
    exit 0
fi

mkdir -p "$UNIT_DIR"

cat > "$UNIT_DIR/memoirs-api.service" <<UNIT
[Unit]
Description=memoirs HTTP API + Web UI
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT}
Environment="MEMOIRS_AUTO_SNAPSHOT=daily"
Environment="MEMOIRS_SNAPSHOT_KEEP=14"
ExecStart=${VENV}/bin/python -c "from memoirs.api.server import run; from pathlib import Path; run(db_path=Path('${DB}'), host='127.0.0.1', port=${PORT})"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now memoirs-api

echo "✓ memoirs-api installed and started"
echo "   → http://127.0.0.1:${PORT}"
echo "   logs:   journalctl --user -u memoirs-api -f"
echo "   status: systemctl --user status memoirs-api"
