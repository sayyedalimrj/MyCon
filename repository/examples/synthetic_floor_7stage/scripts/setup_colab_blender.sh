#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Setup Blender 4.2 LTS on Google Colab and verify the GPU is wired up.
# ---------------------------------------------------------------------
#
# Usage (in a Colab cell):
#
#     !bash examples/synthetic_floor_7stage/scripts/setup_colab_blender.sh
#
# After this script finishes, ``/content/blender/blender`` will be on
# disk and ``--device OPTIX`` will work in run_blender_gpu.py.
#
# Tested on Colab T4 (sm_75) and A100 (sm_80) runtimes with Blender
# 4.2.x portable Linux x64 build.
# ---------------------------------------------------------------------

set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
BLENDER_MAJOR="${BLENDER_VERSION%.*}"           # e.g. 4.2
BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_MAJOR}/${BLENDER_TARBALL}"
INSTALL_DIR="${INSTALL_DIR:-/content/blender}"

echo "== Setup Blender ${BLENDER_VERSION} =="

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. The renderer will fall back to CPU."
else
    nvidia-smi | head -25 || true
fi

# Runtime libraries Blender needs even in headless mode
echo "-- installing system libs (libxrender / libxi / etc.) --"
apt-get -qq install -y --no-install-recommends \
    libxrender1 libxi6 libxkbcommon0 libsm6 libxxf86vm1 libgl1 libegl1 libxfixes3 \
    >/dev/null

if [ ! -x "${INSTALL_DIR}/blender" ]; then
    echo "-- downloading ${BLENDER_TARBALL} --"
    mkdir -p "${INSTALL_DIR}"
    curl -fsSL -o "/tmp/${BLENDER_TARBALL}" "${BLENDER_URL}"
    tar -xf "/tmp/${BLENDER_TARBALL}" -C "${INSTALL_DIR}" --strip-components=1
    rm -f "/tmp/${BLENDER_TARBALL}"
fi

echo "-- blender version --"
"${INSTALL_DIR}/blender" --version | head -5

echo "-- enumerating Cycles devices (OPTIX preferred) --"
"${INSTALL_DIR}/blender" -b --python-expr "
import bpy
prefs = bpy.context.preferences.addons['cycles'].preferences
ok = False
for backend in ('OPTIX', 'CUDA', 'CPU'):
    try:
        prefs.compute_device_type = backend
    except TypeError:
        continue
    prefs.get_devices()
    devs = [d for d in prefs.devices if d.type == backend]
    if devs:
        print(f'available backend: {backend}')
        for d in devs:
            print(f'   {d.name} ({d.type})')
        ok = True
print('GPU READY' if ok else 'NO GPU; will run on CPU')
" 2>/dev/null | tail -20 || true

echo "-- export PATH so 'blender' is callable --"
echo "export PATH=${INSTALL_DIR}:\$PATH" >> ~/.bashrc
export PATH="${INSTALL_DIR}:$PATH"

cat <<EOF

OK. Blender installed at ${INSTALL_DIR}/blender

Next:
    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \\
            --blender ${INSTALL_DIR}/blender --quick

EOF
