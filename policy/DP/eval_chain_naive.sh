#!/bin/bash
# Naive long-horizon eval: PI1 (open) → PI2 (load) → PI3 (close), one DP server per stage.
#
# Kept environment variables (full reference + accepted values: see
# RoboTwin_Cabinet_Objects_SkillChain.md §5.3):
#   run:        TEST_NUM SEED_START TASK_CONFIG POLICY_GPU ENV_GPU VIDEO_SUBSTEP
#               STALL_TIMEOUT_SEC MAX_RESTARTS STOP_AFTER LOAD_EARLY_STOP OUTPUT_ROOT
#   handoff 1:  SWITCH_MODE BRIDGE BRIDGE_ALPHA OPEN_HANDOFF V_MODEL V_FEATURE_SET
#               V_THRESHOLD V_KAPPA V_EXEC V_MIN_OPEN_FRAC OBSTACLES FLAG_SOURCE
#   handoff 2:  LOAD_SWITCH BRIDGE23 V3_MODEL V3_FEATURE_SET V3_THRESHOLD V3_KAPPA
#   perturb:    PERTURB_DRAWER OBJECT_DR PERTURB_EE DRAWER_LIMIT
# Defaults live in the eval script's argparse; only the variables you set are forwarded.
#   DRY_RUN=1  print the argv that would be run and exit (no conda, no sim).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Launcher-only knobs (ENV_GPU → CUDA_VISIBLE_DEVICES; MAX_RESTARTS + the STALL_TIMEOUT_SEC
# fallback below drive the external watchdog. The fallback mirrors argparse's 600 s default,
# and STALL_TIMEOUT_SEC is forwarded to Python only when you set it.).
ENV_GPU="${ENV_GPU:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-10}"

# --- validation: reject values that would otherwise silently misbehave --------
if [[ -n "${LOAD_EARLY_STOP:-}" && "${LOAD_EARLY_STOP}" != "0" && "${LOAD_EARLY_STOP}" != "1" ]]; then
  echo "[CHAIN] LOAD_EARLY_STOP must be 0 or 1 (got '${LOAD_EARLY_STOP}')" >&2; exit 2
fi
case "${LOAD_SWITCH:-flag}" in flag|value) ;; *)
  echo "[CHAIN] LOAD_SWITCH must be flag|value (got '${LOAD_SWITCH:-}')" >&2; exit 2 ;; esac
case "${BRIDGE23:-none}" in none|vsearch) ;; *)
  echo "[CHAIN] BRIDGE23 must be none|vsearch (got '${BRIDGE23:-}')" >&2; exit 2 ;; esac
case "${OBSTACLES:-privileged}" in privileged|perceived) ;; *)
  echo "[CHAIN] OBSTACLES must be privileged|perceived (got '${OBSTACLES:-}')" >&2; exit 2 ;; esac
case "${FLAG_SOURCE:-sim}" in sim|perceived) ;; *)
  echo "[CHAIN] FLAG_SOURCE must be sim|perceived (got '${FLAG_SOURCE:-}')" >&2; exit 2 ;; esac
if [[ "${BRIDGE:-none}" == "partial" && -z "${BRIDGE_ALPHA:-}" ]]; then
  echo "[CHAIN] BRIDGE=partial requires BRIDGE_ALPHA (0..1)" >&2; exit 2
fi
if [[ -n "${BRIDGE_ALPHA:-}" && "${BRIDGE:-none}" != "partial" ]]; then
  echo "[CHAIN] BRIDGE_ALPHA only applies to BRIDGE=partial (got BRIDGE=${BRIDGE:-none})" >&2; exit 2
fi
if [[ "${BRIDGE:-none}" == "vsearch" || "${SWITCH_MODE:-flag}" == "value" ]] && [[ -z "${V_MODEL:-}" ]]; then
  echo "[CHAIN] BRIDGE=vsearch / SWITCH_MODE=value require V_MODEL (checkpoints/.../model.pt)" >&2; exit 2
fi
if [[ "${LOAD_SWITCH:-flag}" == "value" || "${BRIDGE23:-none}" == "vsearch" ]] && [[ -z "${V3_MODEL:-}" ]]; then
  echo "[CHAIN] LOAD_SWITCH=value / BRIDGE23=vsearch require V3_MODEL" >&2; exit 2
fi

# --- ENV_VAR:--flag mapping --------------------------------------------------
# One line per kept variable; only the ones that are set are forwarded, so argparse stays
# the single source of truth for every default. Special cases (LOAD_EARLY_STOP's 0/1 flag,
# PERTURB_DRAWER's two tokens) are appended below.
ARGS=()
while IFS=: read -r var flag; do
  if [[ -n "${var}" && -n "${!var:-}" ]]; then
    ARGS+=("${flag}" "${!var}")
  fi
done <<'MAP'
TASK_CONFIG:--task-config
TEST_NUM:--test-num
SEED_START:--seed-start
POLICY_GPU:--policy-gpu
OUTPUT_ROOT:--output-root
SWITCH_MODE:--switch
BRIDGE:--bridge
BRIDGE_ALPHA:--bridge-alpha
OPEN_HANDOFF:--open-handoff
FLAG_SOURCE:--flag-source
V_MODEL:--v-model
V_THRESHOLD:--v-threshold
V_KAPPA:--v-kappa
V_FEATURE_SET:--v-feature-set
V_EXEC:--v-exec
V_MIN_OPEN_FRAC:--v-min-open-frac
OBSTACLES:--obstacles
STOP_AFTER:--stop-after
VIDEO_SUBSTEP:--video-substep
STALL_TIMEOUT_SEC:--stall-timeout-sec
LOAD_SWITCH:--load-switch
BRIDGE23:--bridge23
V3_MODEL:--v3-model
V3_FEATURE_SET:--v3-feature-set
V3_THRESHOLD:--v3-threshold
V3_KAPPA:--v3-kappa
OBJECT_DR:--object-dr
PERTURB_EE:--perturb-ee
DRAWER_LIMIT:--drawer-limit
MAP
if [[ "${LOAD_EARLY_STOP:-0}" == "1" ]]; then ARGS+=(--load-early-stop); fi
if [[ -n "${PERTURB_DRAWER:-}" ]]; then
  read -r pd_lo pd_hi <<< "${PERTURB_DRAWER}"
  ARGS+=(--perturb-drawer "${pd_lo}" "${pd_hi}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "python scripts/eval_cabinet_chain_naive.py ${ARGS[*]} --output-dir <run-dir>"
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate hooks (e.g. cuda-nvcc) touch unset vars under `set -u`
set +u
conda activate RoboTwin
set -u

cd "${BENCH_ROOT}"
export CUDA_VISIBLE_DEVICES="${ENV_GPU}"
# Per-step progress uses `print(..., end="\r")`; unbuffered keeps it visible in a log.
export PYTHONUNBUFFERED=1
# RTX 5090: OIDN CUDA unsupported → noisy svulkan2 errors / possible stall.
export SAPIEN_RT_DENOISER="${SAPIEN_RT_DENOISER:-none}"

# Fixed output dir, so a restart resumes from the same records.jsonl. The tag depends only
# on these args, so the same argv is used for the print call and the run.
OUTPUT_DIR="$(python scripts/eval_cabinet_chain_naive.py --print-output-dir "${ARGS[@]}")"
echo "[CHAIN] output dir: ${OUTPUT_DIR}"

RUN_ARGS=(scripts/eval_cabinet_chain_naive.py "${ARGS[@]}" --output-dir "${OUTPUT_DIR}")

attempt=0
while true; do
  set +e
  # Fresh heartbeat before launch: a stale one (restart, or a re-run into an existing
  # dir) would make the first check, 10 s in and before Python reaches beat() past the
  # sapien/torch/curobo imports, kill the run and burn a restart.
  touch "${OUTPUT_DIR}/.heartbeat"
  # Own process group (server + ffmpeg included) so a hung run can be killed as a whole.
  setsid python "${RUN_ARGS[@]}" &
  pid=$!
  # External watchdog: a render hang can hold the GIL, so the in-process one may never fire.
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 10
    hb="${OUTPUT_DIR}/.heartbeat"
    [[ -f "${hb}" ]] || continue
    age=$(( $(date +%s) - $(stat -c %Y "${hb}") ))
    if [[ "${STALL_TIMEOUT_SEC:-600}" -gt 0 && "${age}" -gt "${STALL_TIMEOUT_SEC:-600}" ]]; then
      echo "[CHAIN] stall: no heartbeat for ${age}s -> killing pgid ${pid}"
      kill -9 -- "-${pid}" 2>/dev/null
      cur="${OUTPUT_DIR}/.current_episode"
      if [[ -f "${cur}" ]]; then
        # Stub for the in-flight seed: the resume skips it. label=stall so consumers
        # never hit a missing key; the marker is cleared so a later stall cannot write
        # a duplicate line for a seed that already has a record.
        python -c 'import json,sys; r=json.load(open(sys.argv[1])); r["label"]="stall"; r["error"]="stall"; print(json.dumps(r))' \
          "${cur}" >> "${OUTPUT_DIR}/records.jsonl"
        rm -f "${cur}"
      fi
      break
    fi
  done
  wait "${pid}"
  code=$?
  # Always reap the whole process group: a crashed run otherwise leaves its policy server
  # (~1 GB GPU) orphaned, and the leak piles up into OOM / vk::DeviceLostError.
  kill -9 -- "-${pid}" 2>/dev/null || true
  # Killed by the external watchdog -> same restart path as the in-process one (3).
  # 137 = killed by the external watchdog; 134 = abort (vk::DeviceLostError on RTX 5090) -> restart
  [[ "${code}" -eq 137 || "${code}" -eq 134 ]] && code=3
  set -e
  # 3 = stall watchdog fired; anything else is final.
  if [[ "${code}" -ne 3 ]]; then break; fi
  if [[ "${attempt}" -ge "${MAX_RESTARTS}" ]]; then
    echo "[CHAIN] stall: max restarts (${MAX_RESTARTS}) reached"
    break
  fi
  attempt=$((attempt + 1))
  echo "[CHAIN] stall -> restarting (${attempt}/${MAX_RESTARTS})"
done

echo "[CHAIN] eval finished (switch=${SWITCH_MODE:-flag}, exit=${code})"
exit "${code}"
