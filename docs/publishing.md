# Publishing memoirs to PyPI

## One-time setup

### 1. Create a PyPI account
- https://pypi.org/account/register/

### 2. Configure Trusted Publishing on PyPI (recommended — no tokens needed)

Go to https://pypi.org/manage/account/publishing/ and add a "pending publisher":

| field | value |
|---|---|
| PyPI Project Name | `memoirs` |
| Owner | `misaelzapata` |
| Repository name | `memoirs` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

After your first release, it auto-converts to a real publisher; no tokens ever stored on disk or in CI secrets.

### 3. Add the GitHub environment

```bash
gh api -X PUT repos/misaelzapata/memoirs/environments/pypi
```

This unlocks `environment: pypi` in `.github/workflows/release.yml`.

## Releasing 0.1.0

```bash
# 1. Tag the release
git tag -a v0.1.0 -m "memoirs 0.1.0 — initial public release"
git push origin v0.1.0

# 2. The Release workflow runs automatically and:
#    - builds wheel + sdist
#    - uploads to PyPI via Trusted Publishing
#    - attaches wheel/sdist to the GitHub Release
```

You can also trigger the workflow manually:

```bash
gh workflow run release.yml
```

## Manual upload (fallback)

If Trusted Publishing isn't set up yet:

```bash
python -m build                          # produces dist/memoirs-X.Y.Z*
python -m twine upload --repository pypi dist/*
# you'll be prompted for the API token (or it reads ~/.pypirc)
```

## Pre-release checks

Before tagging:

```bash
# 1. Bump version in pyproject.toml
$EDITOR pyproject.toml      # version = "0.1.0" → "0.1.1"

# 2. Build clean
rm -rf dist build *.egg-info
python -m build

# 3. Smoke install in a fresh venv
python3 -m venv /tmp/check-memoirs && source /tmp/check-memoirs/bin/activate
pip install dist/memoirs-*.whl
memoirs --help && memoirs --db /tmp/probe.sqlite status
deactivate && rm -rf /tmp/check-memoirs

# 4. End-to-end validation
bash scripts/validate_fresh_install.sh

# 5. Final twine check
python -m twine check dist/*
```

If all 5 are green, push the tag.

## Versioning

memoirs follows [SemVer](https://semver.org/):

- `0.x.y` — pre-1.0; minor bumps may break SQLite schema (migrations cover the upgrade path).
- `1.0.0` — schema and MCP tool API frozen; only additive changes.
- Breaking schema changes always ship a numbered migration with `up()` and `down()`.

## What gets published

```
memoirs-0.1.0-py3-none-any.whl     ← what `pip install memoirs` downloads (~370 KB)
memoirs-0.1.0.tar.gz               ← source distribution (~2.7 MB, includes README + screenshots)
```

The wheel is pure-Python (`py3-none-any`) — no native extensions, works everywhere Python 3.10+ runs. Heavy deps (sqlite-vec native lib, llama-cpp-python wheels, sentence-transformers weights) are pulled at extras-install or runtime, not embedded.

## After release

```bash
# Verify it's live
pip index versions memoirs
# → 0.1.0

# Try a fresh install
pip install --upgrade memoirs
memoirs status
```
