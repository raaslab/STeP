#!/usr/bin/env bash
# One-time local dev environment setup for STeP.
#
# STeP's own pyproject.toml only declares `openai` -- the actual LIBERO /
# robosuite / MuJoCo / contact-graspnet / OMPL stack lives in a sibling
# `cap-x` checkout that every scripts/run_libero_*.py imports from at
# runtime via sys.path (see step/system_one/libero_scene.py). This script:
#
#   1. Locates (or clones) cap-x as a sibling of STeP.
#   2. Creates cap-x's base `.venv` (uv, Python 3.10) if missing.
#   3. Creates cap-x's `.venv-libero` (uv, Python 3.12) with the `libero`
#      and `contactgraspnet` extras -- this is the venv every LIBERO script
#      here needs active.
#   4. Bridges OMPL into `.venv-libero` from a conda-forge build. OMPL has
#      no prebuilt PyPI wheel, so `uv sync` alone can't get you Python
#      bindings; this vendors the conda-forge `.so`s (built against a
#      newer libstdc++ than most system installs) straight into the venv's
#      `ompl/` package dir, alongside their own libstdc++/libgcc_s/boost so
#      they never touch the system's. Requires `conda` to be on PATH once,
#      just to fetch prebuilt binaries -- nothing conda-related is needed
#      afterwards to run STeP.
#   5. Writes scripts/env.sh with CONTACT_GRASPNET_CHECKPOINT_DIR pointed at
#      the checkpoint that ships inside cap-x's contact_graspnet_pytorch
#      submodule (configs/libero_template.json ships this blank on purpose --
#      without it, every pick silently falls back to a heuristic grasp
#      generator instead of the real model).
#
# Re-running is safe -- each step skips if its result already exists.
#
# Usage:
#   cd STeP && bash scripts/setup_env.sh [--cap-x-root /path/to/cap-x]
set -euo pipefail

STEP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAPX_ROOT="$(dirname "$STEP_ROOT")/cap-x"
PYTHON_LIBERO_VERSION="3.12"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cap-x-root) CAPX_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

echo "[setup] STeP root:  $STEP_ROOT"
echo "[setup] cap-x root: $CAPX_ROOT"

# --- 1. cap-x checkout ---
if [[ -d "$CAPX_ROOT/.git" ]]; then
  echo "[setup] cap-x already present at $CAPX_ROOT, skipping clone"
else
  echo "[setup] cloning cap-x into $CAPX_ROOT"
  git clone --recurse-submodules https://github.com/capgym/cap-x "$CAPX_ROOT"
fi

# --- 2. uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- 3. cap-x base env ---
if [[ -d "$CAPX_ROOT/.venv" ]]; then
  echo "[setup] $CAPX_ROOT/.venv already exists, skipping base install"
else
  echo "[setup] uv sync (base cap-x install)"
  (cd "$CAPX_ROOT" && uv python install 3.10 && uv venv -p 3.10 && uv sync)
fi

# --- 4. .venv-libero ---
LIBERO_VENV="$CAPX_ROOT/.venv-libero"
if [[ -d "$LIBERO_VENV" ]]; then
  echo "[setup] $LIBERO_VENV already exists, skipping venv creation"
else
  echo "[setup] creating $LIBERO_VENV (python $PYTHON_LIBERO_VERSION)"
  (cd "$CAPX_ROOT" && uv venv .venv-libero --python "$PYTHON_LIBERO_VERSION")
fi

echo "[setup] uv sync --active --extra libero --extra contactgraspnet"
# Set VIRTUAL_ENV/PATH ourselves rather than `source .venv-libero/bin/activate`:
# `uv sync --active` trusts $VIRTUAL_ENV literally, and activate scripts bake
# in an absolute path at creation time that goes stale (and silently makes
# uv sync create/populate a venv at the wrong location) if this checkout is
# ever moved.
(cd "$CAPX_ROOT" && VIRTUAL_ENV="$LIBERO_VENV" PATH="$LIBERO_VENV/bin:$PATH" \
  uv sync --active --extra libero --extra contactgraspnet)

VENV_PY="$LIBERO_VENV/bin/python"
VENV_SITE="$LIBERO_VENV/lib/python${PYTHON_LIBERO_VERSION}/site-packages"

# --- 5. OMPL bridge ---
if "$VENV_PY" -c "import ompl" >/dev/null 2>&1; then
  echo "[setup] ompl already importable in .venv-libero, skipping OMPL bridge"
elif ! command -v conda >/dev/null 2>&1; then
  echo "[setup] warning: ompl not importable and conda not found, cannot bridge it automatically." >&2
  echo "[setup]   Install conda and re-run." >&2
else
  echo "[setup] bridging OMPL from a conda-forge build (no prebuilt PyPI wheel exists)"
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | grep -qE '^\s*ompl_bridge\s'; then
    echo "[setup] creating conda env ompl_bridge (conda-forge ompl, python $PYTHON_LIBERO_VERSION)"
    conda create -n ompl_bridge -y -c conda-forge "python=$PYTHON_LIBERO_VERSION" ompl
  fi
  BRIDGE_LIB="$CONDA_BASE/envs/ompl_bridge/lib"
  BRIDGE_SITE="$BRIDGE_LIB/python${PYTHON_LIBERO_VERSION}/site-packages"

  echo "[setup] copying ompl package into .venv-libero"
  rm -rf "$VENV_SITE/ompl"
  cp -r "$BRIDGE_SITE/ompl" "$VENV_SITE/ompl"
  rm -rf "$VENV_SITE/ompl/__pycache__"

  # libompl.so has an $ORIGIN runpath (it looks for its own deps next to
  # itself), so vendor whatever runtime libs it actually needs -- resolved
  # by name via ldd rather than hardcoding version numbers, since conda-forge
  # may bump its boost/libstdc++ pins over time.
  echo "[setup] vendoring runtime libs (libstdc++/libgcc_s/boost) next to libompl.so"
  NEEDED_LIBS="$(ldd "$VENV_SITE"/ompl/libompl.so.* 2>/dev/null \
    | awk '{print $1}' \
    | grep -E '^lib(std)?c\+\+\.so|^libgcc_s\.so|^libboost_serialization\.so' \
    | sort -u)"
  for lib in $NEEDED_LIBS; do
    src="$(find "$BRIDGE_LIB" -maxdepth 1 -name "$lib" -print -quit)"
    if [[ -n "$src" ]]; then
      cp -L "$src" "$VENV_SITE/ompl/$lib"
    else
      echo "[setup] warning: could not find $lib in $BRIDGE_LIB" >&2
    fi
  done

  PRELOAD_PTH="$VENV_SITE/_ompl_libstdcxx_preload.pth"
  echo "[setup] writing $PRELOAD_PTH (forces the bundled libstdc++ to load before ompl's extension does)"
  cat > "$PRELOAD_PTH" <<EOF
import ctypes, os; _p = "$VENV_SITE/ompl/libstdc++.so.6"; os.path.exists(_p) and ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
EOF

  if "$VENV_PY" -c "import ompl" >/dev/null 2>&1; then
    echo "[setup] OMPL bridge OK"
  else
    echo "[setup] warning: ompl still fails to import after bridging." >&2
  fi
fi

# --- 6. env.sh (contact-graspnet checkpoint) ---
CKPT_DIR="$CAPX_ROOT/capx/third_party/contact_graspnet_pytorch/checkpoints/contact_graspnet"
if [[ -d "$CKPT_DIR" ]]; then
  echo "export CONTACT_GRASPNET_CHECKPOINT_DIR=\"$CKPT_DIR\"" > "$STEP_ROOT/scripts/env.sh"
  echo "[setup] wrote $STEP_ROOT/scripts/env.sh -- source it before running anything"
else
  echo "[setup] warning: expected checkpoint dir missing: $CKPT_DIR" >&2
  echo "[setup]   picks will silently fall back to the heuristic grasp generator without it." >&2
fi

echo ""
echo "[setup] done. Sanity check:"
echo "  source $LIBERO_VENV/bin/activate"
echo "  source $STEP_ROOT/scripts/env.sh"
echo "  export OPENAI_API_KEY=..."
echo "  cd $STEP_ROOT"
echo "  python -c \"from libero.benchmark import get_benchmark; print(get_benchmark('libero_spatial')().n_tasks)\""
