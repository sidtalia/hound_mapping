#!/usr/bin/env bash
# Generate multicam rail dataset (if missing) and run smoke_rail_mapper.
#
# Example:
#   bash scripts/smoke_multicam_rail.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ELEV="${ELEV:-${ROOT}/data/rail_sim/race-2_height_0p05.npy}"
# Prefer IGHA path if present next to elev; else require PATH_NPY.
PATH_NPY="${PATH_NPY:-}"
OUT="${OUT:-${ROOT}/data/rail_sim/race-2_multicam}"
MAX_FRAMES="${MAX_FRAMES:-40}"

if [[ ! -f "${ELEV}" ]]; then
  echo "Missing elev: ${ELEV}" >&2
  echo "Set ELEV=... to a heightfield .npy" >&2
  exit 1
fi

if [[ -z "${PATH_NPY}" ]]; then
  # Common IGHAStar map path location (optional).
  CAND=(
    /root/colcon_ws/src/IGHAStar/Maps/Offroad/race-2_kinodynamic_elevation_IGHAStar_path.npy
    "${ROOT}/data/rail_sim/race-2_path.npy"
  )
  for c in "${CAND[@]}"; do
    if [[ -f "${c}" ]]; then
      PATH_NPY="${c}"
      break
    fi
  done
fi
if [[ -z "${PATH_NPY}" || ! -f "${PATH_NPY}" ]]; then
  echo "Set PATH_NPY=... to an IGHA* path .npy" >&2
  exit 1
fi

if [[ ! -f "${OUT}/camera.txt" ]]; then
  echo "=== Generating multicam dataset → ${OUT} ==="
  python3 "${ROOT}/scripts/generate_rail_dataset.py" \
    --elev "${ELEV}" \
    --path "${PATH_NPY}" \
    --out "${OUT}" \
    --multicam \
    --max-frames "${MAX_FRAMES}" \
    --voxel-size 0.1 \
    --window-radius 10.0
fi

echo "=== smoke_rail_mapper multicam ==="
source /opt/ros/jazzy/setup.bash
source /root/colcon_ws/install/setup.bash 2>/dev/null || true
ros2 run hound_mapping smoke_rail_mapper --ros-args \
  -p dataset:="${OUT}" \
  -p max_frames:="${MAX_FRAMES}"
