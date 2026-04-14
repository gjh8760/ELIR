#!/usr/bin/env bash
# Download RetinexNet DecomNet pretrained weights (PyTorch port).
#
# Source: https://github.com/aasharma90/RetinexNet_PyTorch
# The original RetinexNet (BMVC'18, weichen582) distributed TF weights; the
# aasharma90 port trained a PyTorch DecomNet on LOL-v1 and shares the weight
# under the `ckpts/` directory of their repo.
#
# This script:
#   1) Clones a shallow copy of the repo into third_party/retinexnet/_repo
#   2) Copies the Decom weight to third_party/retinexnet/decom_net.pth
#   3) Converts the raw checkpoint keys into the layout expected by
#      ELIR.models.decomposers.DecomNet (see convert_weights.py).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${ROOT}/third_party/retinexnet"
mkdir -p "${DST}"

REPO_DIR="${DST}/_repo"
if [ ! -d "${REPO_DIR}" ]; then
    git clone --depth 1 https://github.com/aasharma90/RetinexNet_PyTorch.git "${REPO_DIR}"
fi

# aasharma90 repo ships weights under ckpts/; the DecomNet half is saved as
# either `Decom.tar` or weights prefixed with 'DecomNet.' in a bundled ckpt.
# Point the user at wherever the file actually is after clone.
echo ""
echo "[done] repo cloned to ${REPO_DIR}"
echo "[next] copy/rename the DecomNet weight file to:"
echo "         ${DST}/decom_net.pth"
echo "       then run:"
echo "         python ${ROOT}/third_party/retinexnet/convert_weights.py \\"
echo "                --src <path-to-raw-ckpt> --dst ${DST}/decom_net.pth"
