#!/bin/bash
# Phase 2a handoff probe: random left-arm handoff state -> successor DP label.
# SKILL=load|close  N=episodes  SEED_START=first seed  OUTPUT_DIR=result dir (required)
# SAMPLER=box|release (release = pi1's measured mean release pose; load/left only)
# POLICY_GPU=GPU id for the DP server (ENV_GPU selects the sim GPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SKILL="${SKILL:-load}"
N="${N:-10}"
SEED_START="${SEED_START:-500000}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-0}"
TASK_CONFIG="${TASK_CONFIG:-cabinet_chain_nominal_dr}"
STEP_BUDGET="${STEP_BUDGET:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
# ARM: left|right (default: left for load, right for close — see handoff_probe.py).
ARM="${ARM:-}"
# DRAWER_FRAC_RANGE: "LO HI" handoff drawer fraction (default: load 0.5 0.9).
DRAWER_FRAC_RANGE="${DRAWER_FRAC_RANGE:-}"
# PLAN_ONLY=1: setup + sampling + move only, no policy server (dry run).
PLAN_ONLY="${PLAN_ONLY:-0}"
# PLAN_ATTEMPTS: resample a target the planner refuses (default 1 = no retry).
PLAN_ATTEMPTS="${PLAN_ATTEMPTS:-1}"
# SAMPLER: box (default) | release (pi1's measured mean release pose, load/left only).
SAMPLER="${SAMPLER:-box}"

if [[ "${SKILL}" != "load" && "${SKILL}" != "close" ]]; then
  echo "[PROBE] SKILL must be load or close (got '${SKILL}')" >&2
  exit 2
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "[PROBE] OUTPUT_DIR is required" >&2
  exit 2
fi
# SAPIEN render hangs: kill after this many seconds without progress, then resume.
STALL_TIMEOUT_SEC="${STALL_TIMEOUT_SEC:-600}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate hooks (e.g. cuda-nvcc) touch unset vars under `set -u`
set +u
conda activate RoboTwin
set -u

cd "${BENCH_ROOT}"
export CUDA_VISIBLE_DEVICES="${ENV_GPU}"
# Per-step progress uses `print(..., end="\r")`; unbuffered keeps it visible in a log.
export PYTHONUNBUFFERED=1
# RTX 5090: OIDN CUDA unsupported -> noisy svulkan2 errors / possible stall.
export SAPIEN_RT_DENOISER="${SAPIEN_RT_DENOISER:-none}"

mkdir -p "${OUTPUT_DIR}"
echo "[PROBE] skill=${SKILL} n=${N} seed_start=${SEED_START} output=${OUTPUT_DIR} arm=${ARM:-default} plan_only=${PLAN_ONLY} plan_attempts=${PLAN_ATTEMPTS} sampler=${SAMPLER}"

RUN_ARGS=(
  scripts/handoff_probe.py
  --skill "${SKILL}"
  --n "${N}"
  --seed-start "${SEED_START}"
  --output-dir "${OUTPUT_DIR}"
  --task-config "${TASK_CONFIG}"
  --policy-gpu "${POLICY_GPU}"
  --step-budget "${STEP_BUDGET}"
  --plan-attempts "${PLAN_ATTEMPTS}"
  --sampler "${SAMPLER}"
)
[[ -n "${ARM}" ]] && RUN_ARGS+=(--arm "${ARM}")
# Unquoted on purpose: DRAWER_FRAC_RANGE="0.5 0.9" must split into two argv entries.
[[ -n "${DRAWER_FRAC_RANGE}" ]] && RUN_ARGS+=(--drawer-frac-range ${DRAWER_FRAC_RANGE})
[[ "${PLAN_ONLY}" == "1" ]] && RUN_ARGS+=(--plan-only)

attempt=0
code=0
while true; do
  set +e
  # Fresh heartbeat before launch: a stale one (restart, or a re-run into an existing
  # dir) would make the first check, 10 s in and before Python reaches beat() past the
  # sapien/torch/curobo imports, kill the run and burn a restart.
  touch "${OUTPUT_DIR}/.heartbeat"
  # Own process group (policy server included) so a hung run can be killed as a whole.
  setsid python "${RUN_ARGS[@]}" &
  pid=$!
  # External watchdog: a render hang can hold the GIL, so an in-process one may never fire.
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 10
    hb="${OUTPUT_DIR}/.heartbeat"
    [[ -f "${hb}" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "${hb}") ))
    if [[ "${STALL_TIMEOUT_SEC}" -gt 0 && "${age}" -gt "${STALL_TIMEOUT_SEC}" ]]; then
      echo "[PROBE] stall: no heartbeat for ${age}s -> killing pgid ${pid}"
      kill -9 -- "-${pid}" 2>/dev/null
      cur="${OUTPUT_DIR}/.current_episode"
      if [[ -f "${cur}" ]]; then
        # Stub for the in-flight seed: the probe's resume skips it. label=stall so
        # consumers never hit a missing key, and the marker is cleared so a later
        # stall (e.g. during server teardown) cannot write a duplicate line.
        python -c 'import json,sys; r=json.load(open(sys.argv[1])); r["label"]="stall"; r["error"]="stall"; print(json.dumps(r))' \
          "${cur}" >> "${OUTPUT_DIR}/probe.jsonl"
        rm -f "${cur}"
      fi
      break
    fi
  done
  wait "${pid}"
  code=$?
  # Killed by the external watchdog -> same restart path as the in-process one (3).
  [[ "${code}" -eq 137 ]] && code=3
  set -e
  # 3 = stall watchdog fired; anything else is final.
  if [[ "${code}" -ne 3 ]]; then break; fi
  if [[ "${attempt}" -ge "${MAX_RESTARTS}" ]]; then
    echo "[PROBE] stall: max restarts (${MAX_RESTARTS}) reached"
    break
  fi
  attempt=$((attempt + 1))
  echo "[PROBE] stall -> restarting (${attempt}/${MAX_RESTARTS})"
done

echo "[PROBE] finished (skill=${SKILL}, exit=${code})"
exit "${code}"
