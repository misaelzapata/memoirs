#!/usr/bin/env sh
# Memoirs installer — POSIX shell, idempotent.
# Usage:
#   curl -sSL https://raw.githubusercontent.com/<org>/memoirs/main/install.sh | sh
# or:
#   ./install.sh
set -eu

# --- helpers -----------------------------------------------------------------
say()  { printf "\033[1;36m▸\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# --- check prerequisites ----------------------------------------------------
say "checking prerequisites"
command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.10+."

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
PY_MAJOR=${PY_VER%.*}
PY_MINOR=${PY_VER#*.}
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "Python ${PY_VER} found; memoirs requires 3.10+"
fi
say "python ${PY_VER} ✓"

command -v git >/dev/null 2>&1 || warn "git not found — install will skip cloning"

# --- locate or clone repo ---------------------------------------------------
REPO_URL="${MEMOIRS_REPO:-https://github.com/example/memoirs.git}"
TARGET_DIR="${MEMOIRS_DIR:-$HOME/memoirs}"

if [ -f "./pyproject.toml" ] && grep -q "name = \"memoirs\"" pyproject.toml 2>/dev/null; then
    say "found memoirs repo at $(pwd) — using this"
    TARGET_DIR=$(pwd)
elif [ -d "$TARGET_DIR/.git" ] && [ -f "$TARGET_DIR/pyproject.toml" ]; then
    say "found existing memoirs at $TARGET_DIR — pulling"
    (cd "$TARGET_DIR" && git pull --ff-only) || warn "git pull failed; continuing with current code"
else
    if ! command -v git >/dev/null 2>&1; then
        die "git not installed and no local repo found. Install git or clone manually to $TARGET_DIR"
    fi
    say "cloning $REPO_URL into $TARGET_DIR"
    git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

# --- create venv ------------------------------------------------------------
if [ ! -d ".venv" ]; then
    say "creating virtualenv at .venv"
    python3 -m venv .venv
fi
say "activating .venv"
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip

# --- install package + extras -----------------------------------------------
say "installing memoirs (with extras)"
pip install --quiet -e ".[all]" || {
    warn "pip install with [all] failed; falling back to base only"
    pip install --quiet -e .
}

# --- run memoirs setup (interactive unless MEMOIRS_YES=1) -------------------
SETUP_FLAGS=""
if [ "${MEMOIRS_YES:-}" = "1" ] || [ ! -t 0 ]; then
    SETUP_FLAGS="--yes"
    say "non-interactive mode (set MEMOIRS_YES=0 for prompts)"
fi
if [ "${MEMOIRS_SKIP_GEMMA:-}" = "1" ]; then
    SETUP_FLAGS="$SETUP_FLAGS --skip-gemma"
fi

say "running memoirs setup"
memoirs setup $SETUP_FLAGS || warn "setup reported issues — see output above"

# --- final report -----------------------------------------------------------
echo
say "memoirs installed at $TARGET_DIR"
say "to use:"
printf "    cd %s && source .venv/bin/activate\n" "$TARGET_DIR"
printf "    memoirs daemon start\n"
printf "    memoirs ask \"qué prefiere el usuario\"\n"
echo
say "MCP server is configured for any clients you have (Claude Code, Codex, VS Code)"
say "see STATUS.md for diagnostics, GAP.md for the roadmap"
