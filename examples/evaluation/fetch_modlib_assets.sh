#!/usr/bin/env bash
#
# Copyright 2026 Sony  Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Fetch sample assets from modlib (get_assets.py) into tools/evaluation/examples.
# Run from repo root or from tools/evaluation. Requires: git, uv.

set -euo pipefail

MODLIB_REPO="https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git"
MODLIB_BRANCH="main"
MODLIB_MIN_VERSION="1.4"  # script only works with this layout/API

# Resolve tools/evaluation directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
ASSETS_DIR="${EXAMPLES_DIR}/assets"
CLONE_DIR="$(mktemp -d)"
trap 'rm -rf "$CLONE_DIR"' EXIT

echo "Cloning modlib (branch ${MODLIB_BRANCH}) into ${CLONE_DIR} ..."
git clone --depth 1 --branch "$MODLIB_BRANCH" "$MODLIB_REPO" "$CLONE_DIR"
cd "$CLONE_DIR"

# Require modlib >= MODLIB_MIN_VERSION
REQ_MAJOR=$(echo "$MODLIB_MIN_VERSION" | cut -d. -f1); REQ_MAJOR=${REQ_MAJOR:-0}
REQ_MINOR=$(echo "$MODLIB_MIN_VERSION" | cut -d. -f2); REQ_MINOR=${REQ_MINOR:-0}
MODLIB_VERSION=""
if [ -f "pyproject.toml" ]; then
  MODLIB_VERSION=$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*["'\'']\([^"'\'']*\)["'\''].*/\1/p' pyproject.toml | head -1)
fi
if [ -z "$MODLIB_VERSION" ]; then
  git fetch --tags origin 2>/dev/null || true
  DESCRIBE=$(git describe --tags HEAD 2>/dev/null || true)
  if [ -n "$DESCRIBE" ]; then
    MODLIB_VERSION=$(echo "$DESCRIBE" | sed -E 's/^v//; s/-.*//')
  fi
fi
if [ -n "$MODLIB_VERSION" ]; then
  MAJOR=$(echo "$MODLIB_VERSION" | cut -d. -f1); MAJOR=${MAJOR:-0}
  MINOR=$(echo "$MODLIB_VERSION" | cut -d. -f2); MINOR=${MINOR:-0}
  if [ "$MAJOR" -lt "$REQ_MAJOR" ] || { [ "$MAJOR" -eq "$REQ_MAJOR" ] && [ "$MINOR" -lt "$REQ_MINOR" ]; }; then
    echo "modlib version ${MODLIB_VERSION} is below ${MODLIB_MIN_VERSION}; this script requires modlib >= ${MODLIB_MIN_VERSION}." >&2
    exit 1
  fi
  echo "modlib version ${MODLIB_VERSION} (>= ${MODLIB_MIN_VERSION}, ok)"
else
  echo "Could not determine modlib version; this script requires modlib >= ${MODLIB_MIN_VERSION}. Aborting." >&2
  exit 1
fi

echo "Installing modlib in a temporary venv ..."
uv venv .venv
uv sync

echo "Running examples/get_assets.py ..."
if [ -f "examples/get_assets.py" ]; then
  # If get_assets accepts an output dir, pass it; otherwise it may create examples/assets in clone
  if uv run python examples/get_assets.py --help 2>/dev/null | grep -q "output\|dir\|path"; then
    uv run python examples/get_assets.py --output-dir "$ASSETS_DIR" 2>/dev/null || \
    uv run python examples/get_assets.py --out "$ASSETS_DIR" 2>/dev/null || \
    uv run python examples/get_assets.py "$ASSETS_DIR" 2>/dev/null || \
    uv run python examples/get_assets.py
  else
    uv run python examples/get_assets.py
  fi
else
  echo "examples/get_assets.py not found in clone. Listing examples/:" >&2
  ls -la examples/ 2>/dev/null || true
  exit 1
fi

# Copy assets into evaluation/examples if get_assets wrote them inside the clone
if [ -d "${CLONE_DIR}/examples/assets" ] && [ "${CLONE_DIR}/examples/assets" != "$ASSETS_DIR" ]; then
  echo "Copying ${CLONE_DIR}/examples/assets -> ${ASSETS_DIR}"
  mkdir -p "$EXAMPLES_DIR"
  cp -R "${CLONE_DIR}/examples/assets" "$ASSETS_DIR"
fi

if [ -d "$ASSETS_DIR" ]; then
  echo "Assets are in ${ASSETS_DIR}"
  ls -la "$ASSETS_DIR"
else
  echo "No ${ASSETS_DIR} found. get_assets.py may use a different layout; check modlib examples." >&2
  exit 1
fi
