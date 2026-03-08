#!/usr/bin/env bash
# setup_kraken_env.sh
# Creates an isolated Python venv for Kraken inside tjk_suetterlin/kraken_env/
# Safe to re-run (idempotent): skips creation if kraken_env/ already exists.
#
# Usage:
#   bash setup_kraken_env.sh
#
# The script tries uv first (fast), falls back to standard venv + pip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/kraken_env"

echo "=== Kraken Isolated Environment Setup ==="
echo "Target: ${VENV_DIR}"
echo ""

# ── Step 1: Skip if already exists and kraken is importable ─────────────────
if [ -d "${VENV_DIR}" ] && [ -f "${VENV_DIR}/bin/python" ]; then
    echo "✓ kraken_env already exists. Verifying Kraken install..."
    if "${VENV_DIR}/bin/python" -c "from importlib.metadata import version; print('  kraken', version('kraken'))" 2>/dev/null; then
        echo "✓ Kraken is ready. Nothing to do."
        echo ""
        echo "Interpreter: ${VENV_DIR}/bin/python"
        echo "=== Setup complete (already done) ==="
        exit 0
    else
        echo "⚠ kraken_env exists but Kraken import failed. Reinstalling..."
    fi
fi

# ── Step 2: Ensure uv is available ──────────────────────────────────────────
USE_UV=false

if command -v uv &>/dev/null; then
    echo "✓ Found uv: $(uv --version)"
    USE_UV=true
else
    echo "→ uv not found. Attempting to install via official installer..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || true
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh || true
    else
        echo "⚠ Neither curl nor wget available. Cannot install uv."
    fi

    # Add common uv install locations to PATH for this session
    export PATH="${HOME}/.cargo/bin:${HOME}/.local/bin:${PATH}"

    if command -v uv &>/dev/null; then
        echo "✓ uv installed successfully: $(uv --version)"
        USE_UV=true
    else
        echo "⚠ uv install failed or not in PATH. Falling back to standard venv + pip."
        USE_UV=false
    fi
fi

# ── Step 3: Create venv ──────────────────────────────────────────────────────
echo ""
echo "→ Creating virtual environment at ${VENV_DIR}..."

if [ "${USE_UV}" = true ]; then
    # Try Python 3.10 first (recommended for Kraken), fall back to python3
    if uv venv "${VENV_DIR}" --python python3.10 2>/dev/null; then
        echo "✓ Created venv with Python 3.10 (via uv)"
    elif uv venv "${VENV_DIR}" --python python3 2>/dev/null; then
        echo "✓ Created venv with python3 (via uv)"
    else
        echo "⚠ uv venv failed. Falling back to python3 -m venv..."
        python3 -m venv "${VENV_DIR}"
        echo "✓ Created venv with python3 -m venv"
        USE_UV=false
    fi
else
    python3 -m venv "${VENV_DIR}"
    echo "✓ Created venv with python3 -m venv"
fi

# ── Step 4: Install Kraken ───────────────────────────────────────────────────
echo ""
echo "→ Installing kraken (this may take several minutes on first run)..."

if [ "${USE_UV}" = true ]; then
    uv pip install --python "${VENV_DIR}/bin/python" kraken
else
    echo "→ Upgrading pip first..."
    "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
    "${VENV_DIR}/bin/pip" install kraken
fi

# ── Step 5: Patch kraken/blla.py for Python importlib.resources compatibility ─
# kraken 6.x uses resources.files(__name__) inside blla.py (a module, not a
# package), which raises TypeError: 'kraken.blla' is not a package on Python
# 3.10+.  The fix is to use resources.files('kraken') instead, since blla.mlmodel
# lives in the kraken/ package directory.
echo ""
echo "→ Patching kraken/blla.py for importlib.resources compatibility..."

BLLA_PY=$("${VENV_DIR}/bin/python" -c "import kraken.blla; print(kraken.blla.__file__)" 2>/dev/null)
if [ -n "${BLLA_PY}" ]; then
    if grep -q "resources.files(__name__)" "${BLLA_PY}"; then
        sed -i "s/resources\.files(__name__)/resources.files('kraken')/g" "${BLLA_PY}"
        # Remove stale bytecode cache so Python picks up the patched source
        BLLA_PYC="${BLLA_PY%%.py}.cpython-$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")').pyc"
        BLLA_PYCACHE="$(dirname "${BLLA_PY}")/__pycache__/$(basename "${BLLA_PY%%.py}").cpython-"*".pyc"
        rm -f ${BLLA_PYCACHE} 2>/dev/null || true
        echo "✓ Patched: resources.files(__name__) → resources.files('kraken') in ${BLLA_PY}"
    else
        echo "✓ blla.py already patched or uses different API — no change needed."
    fi
else
    echo "⚠ Could not locate kraken/blla.py — skipping patch."
fi

# ── Step 6: Verify installation ──────────────────────────────────────────────
echo ""
echo "→ Verifying installation..."

if KRAKEN_VERSION=$("${VENV_DIR}/bin/python" -c "from importlib.metadata import version; print(version('kraken'))" 2>&1); then
    echo "✓ Kraken ${KRAKEN_VERSION} installed successfully."
    echo ""
    echo "Interpreter: ${VENV_DIR}/bin/python"
    echo "=== Setup complete ==="
    exit 0
else
    echo "✗ ERROR: Kraken installation verification failed."
    echo "  Output: ${KRAKEN_VERSION}"
    echo "  Please check the output above for errors and try again."
    exit 1
fi
