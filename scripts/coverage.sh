#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/coverage run -m pytest tests/ --tb=no -q "$@"
.venv/bin/coverage report --show-missing
.venv/bin/coverage html -d .coverage_html
echo "HTML report: file://$(pwd)/.coverage_html/index.html"
