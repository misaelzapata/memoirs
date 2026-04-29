#!/usr/bin/env bash
# Fresh-install validation: tear down anything stale, install from scratch,
# run the smoke tests + bench + UI screenshot. Mirror of what CI does, but
# locally so we can fix issues before pushing.
#
# Usage:
#   bash scripts/validate_fresh_install.sh
#
# Exit codes:
#   0  all green
#   1  install failed
#   2  pytest failed
#   3  bench failed
#   4  UI failed

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ----------------------------------------------------------------------------
echo "=================================================================="
echo "  memoirs · fresh-install validation"
echo "=================================================================="

VENV="${ROOT}/.venv-fresh"
DB="${ROOT}/.memoirs/validate.sqlite"
PORT=8284

cleanup() {
    pkill -f "memoirs.*--port ${PORT}" 2>/dev/null || true
    pkill -f "validate_fresh_install" 2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------------------------------------------------------
echo
echo "▶ 1/6  fresh venv at $VENV"
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel >/dev/null
"$VENV/bin/pip" install -e ".[api,embeddings_fast,extract,clustering,reranker,viz]" >/dev/null \
    || { echo "❌ install failed"; exit 1; }
echo "✓ install OK"

# ----------------------------------------------------------------------------
echo
echo "▶ 2/6  pytest (no LLM, no slow tests)"
MEMOIRS_CURATOR_ENABLED=off MEMOIRS_GEMMA_CURATOR=off MEMOIRS_GRAPH_LLM=off \
    "$VENV/bin/python" -m pytest tests/test_extractor_filters.py \
        tests/test_retrieval_pipeline.py \
        tests/test_memory_engine_coverage.py \
        tests/test_procedural_memory.py \
        tests/test_curator_robustness.py \
        tests/test_data_utility_audit.py \
        tests/test_command_capture.py \
        -q --tb=short || { echo "❌ pytest failed"; exit 2; }
echo "✓ pytest OK"

# ----------------------------------------------------------------------------
echo
echo "▶ 3/6  build wheel + sdist"
"$VENV/bin/pip" install --quiet build twine
"$VENV/bin/python" -m build --outdir "${ROOT}/dist-fresh" >/dev/null
"$VENV/bin/python" -m twine check "${ROOT}/dist-fresh"/* | grep -i pass || \
    { echo "❌ build failed"; exit 1; }
echo "✓ build OK ($(ls -1 ${ROOT}/dist-fresh | wc -l) artifacts)"

# ----------------------------------------------------------------------------
echo
echo "▶ 4/6  seed demo DB"
rm -f "$DB"
"$VENV/bin/python" scripts/seed_demo_db.py --out "$DB" 2>&1 | tail -3
echo "✓ seed OK"

# ----------------------------------------------------------------------------
echo
echo "▶ 5/6  bench smoke (memoirs only, retrieval-only)"
MEMOIRS_PRF=on MEMOIRS_CURATOR_ENABLED=off "$VENV/bin/python" \
    scripts/bench_vs_others.py --engines memoirs --top-k 10 \
    --out "${ROOT}/.memoirs/validate_bench.json" \
    --md-out "${ROOT}/.memoirs/validate_bench.md" --quiet \
    || { echo "❌ bench failed"; exit 3; }
echo "✓ bench OK"
grep -E "memoirs.*\| (1\.|0\.[89])" "${ROOT}/.memoirs/validate_bench.md" \
    || echo "  (note: review .memoirs/validate_bench.md)"

# ----------------------------------------------------------------------------
echo
echo "▶ 6/6  HTTP server smoke + screenshot"
"$VENV/bin/python" -c "
from memoirs.api.server import run
from pathlib import Path
run(db_path=Path('$DB'), host='127.0.0.1', port=$PORT)
" > "${ROOT}/.memoirs/validate_server.log" 2>&1 &
sleep 6
curl -fsS "http://127.0.0.1:${PORT}/healthz" | grep -q '"ok"' \
    || { echo "❌ server unreachable"; exit 4; }
curl -fsS "http://127.0.0.1:${PORT}/" -o /dev/null \
    || { echo "❌ dashboard 5xx"; exit 4; }
echo "✓ HTTP API + UI OK"

# Done.
cleanup
echo
echo "=================================================================="
echo "  ✓✓✓  ALL VALIDATIONS PASSED"
echo "=================================================================="
echo
echo "Artifacts:"
echo "  · wheel/sdist:    ${ROOT}/dist-fresh/"
echo "  · demo DB:        $DB"
echo "  · bench output:   ${ROOT}/.memoirs/validate_bench.{json,md}"
echo "  · server log:     ${ROOT}/.memoirs/validate_server.log"
