#!/bin/bash
# Naive long-horizon eval: PI1 (open) → PI2 (load) → PI3 (close).
# SWITCH_MODE=flag|mock_vlm|vlm|value  (default flag = B1; value = B3: the open stage ends on
#                                      the V ensemble's best handoff candidate and needs V_MODEL)
# BRIDGE=none|scripted|retreat|partial|vsearch|release_only  (default none; release_only = open gripper only,
#                                        the TFF-only ablation: with SWITCH_MODE=value V scores the current pose; scripted = release + 6cm back-off + home;
#                                        retreat = no home; partial = scripted stopped BRIDGE_ALPHA of the way
#                                        home, needs BRIDGE_ALPHA; vsearch = value-guided handoff search
#                                        (escape the drawer collision, then try the best reachable
#                                        candidate), needs V_MODEL, optional V_THRESHOLD / V_KAPPA /
#                                        V_EXEC=direct_first|two_phase)
# Phase-1 experiments: OPEN_HANDOFF=frac|release, STEER_K=K (0=off), STEER_CHUNKS=5,
#                      STOP_AFTER=close|load, LOAD_EARLY_STOP=0|1,
#                      TRANSITION_WINDOW=W (pi1 calls after the release handoff; needs release)
# Phase-3a perturbations (all off by default): PERTURB_DRAWER="LO HI" (redraw the drawer to
#                      U[LO,HI] after pi1's open stage), OBJECT_DR=load (build the objects with
#                      the load-stage DR), PERTURB_EE=CM (offset the left EE by U[-CM,CM]^3 cm)
# Phase-3b: DRAWER_LIMIT=F (cap the drawer's travel at F of its range; F < 0.85 makes the
#                      flag handoff unreachable)
# Obstacles: OBSTACLES=privileged|perceived (default privileged = the simulator's cabinet
#                      link AABBs; perceived = boxes merged from the head-camera depth cloud,
#                      scripts/perceived_obstacles.py; adds "-pcd" to the run tag)
# Phase-5 per-handoff switching: LOAD_SWITCH=flag|value (value ends the load stage on V3's
#                      lower bound, needs V3_MODEL), BRIDGE23=none|vsearch (the value-guided
#                      pi2->pi3 right-arm repair, needs V3_MODEL), V3_MODEL=path to the V3
#                      model.pt (scripts/train_handoff_value.py --skill close --feature-set
#                      ee_obj --arm right), optional V3_THRESHOLD / V3_KAPPA (default to
#                      V_THRESHOLD / V_KAPPA).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TEST_NUM="${TEST_NUM:-3}"
TASK_CONFIG="${TASK_CONFIG:-cabinet_chain_nominal_dr}"
SEED_START="${SEED_START:-100000}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-0}"
SWITCH_MODE="${SWITCH_MODE:-flag}"
BRIDGE="${BRIDGE:-none}"
BRIDGE_ALPHA="${BRIDGE_ALPHA:-}"
# BRIDGE=vsearch: path to the handoff value ensemble (scripts/train_handoff_value.py).
V_MODEL="${V_MODEL:-}"
V_THRESHOLD="${V_THRESHOLD:-0.9}"
V_KAPPA="${V_KAPPA:-2.0}"
# direct_first (default): one cuboid-on plan straight to each candidate, two-phase move only
# as a fallback; two_phase = old behaviour (always the 6 cm pre-pose detour).
V_EXEC="${V_EXEC:-direct_first}"
SWITCH_EVERY_N="${SWITCH_EVERY_N:-8}"
MIN_STEPS_BEFORE_SWITCH="${MIN_STEPS_BEFORE_SWITCH:-0}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
VLM_CONDA_ENV="${VLM_CONDA_ENV:-xpl_vlm}"
# Same GPU as DP by default; set VLM_DEVICE=cpu if OOM.
VLM_DEVICE="${VLM_DEVICE:-cuda:${POLICY_GPU}}"
VIDEO_SUBSTEP="${VIDEO_SUBSTEP:-0}"
OPEN_HANDOFF="${OPEN_HANDOFF:-frac}"
STEER_K="${STEER_K:-0}"
STEER_CHUNKS="${STEER_CHUNKS:-5}"
STOP_AFTER="${STOP_AFTER:-close}"
LOAD_EARLY_STOP="${LOAD_EARLY_STOP:-0}"
TRANSITION_WINDOW="${TRANSITION_WINDOW:-0}"
# Phase-3a perturbations ("" = off).
PERTURB_DRAWER="${PERTURB_DRAWER:-}"
OBJECT_DR="${OBJECT_DR:-}"
PERTURB_EE="${PERTURB_EE:-}"
# Phase-3b drawer travel cap ("" = uncapped).
DRAWER_LIMIT="${DRAWER_LIMIT:-}"
# Planner obstacle source (step 3); changes the run tag, so it goes into both calls.
OBSTACLES="${OBSTACLES:-privileged}"
# Phase-5 per-handoff switching (defaults keep the pre-existing behaviour/tag).
LOAD_SWITCH="${LOAD_SWITCH:-flag}"
BRIDGE23="${BRIDGE23:-none}"
V3_MODEL="${V3_MODEL:-}"
# Anything but 0/1 would silently evaluate as false ("true", "yes"), so reject it.
if [[ "${LOAD_EARLY_STOP}" != "0" && "${LOAD_EARLY_STOP}" != "1" ]]; then
  echo "[CHAIN] LOAD_EARLY_STOP must be 0 or 1 (got '${LOAD_EARLY_STOP}')" >&2
  exit 2
fi
# Bridge args are shared by the --print-output-dir call and the run, so the resume dir
# and the run can never disagree on the tag.
BRIDGE_ARGS=(--bridge "${BRIDGE}")
if [[ "${BRIDGE}" == "partial" ]]; then
  if [[ -z "${BRIDGE_ALPHA}" ]]; then
    echo "[CHAIN] BRIDGE=partial requires BRIDGE_ALPHA (0..1)" >&2
    exit 2
  fi
  BRIDGE_ARGS+=(--bridge-alpha "${BRIDGE_ALPHA}")
elif [[ "${BRIDGE}" == "vsearch" ]]; then
  if [[ -z "${V_MODEL}" ]]; then
    echo "[CHAIN] BRIDGE=vsearch requires V_MODEL (checkpoints/.../model.pt)" >&2
    exit 2
  fi
  BRIDGE_ARGS+=(--v-model "${V_MODEL}" --v-threshold "${V_THRESHOLD}" --v-kappa "${V_KAPPA}"
                --v-exec "${V_EXEC}")
elif [[ -n "${BRIDGE_ALPHA}" ]]; then
  # Same guard as the eval script's parser: a stray alpha must not be dropped silently.
  echo "[CHAIN] BRIDGE_ALPHA only applies to BRIDGE=partial (got BRIDGE=${BRIDGE})" >&2
  exit 2
fi
# Phase-1 extras that also change the run tag, so the resume dir matches the run.
PHASE1_ARGS=(
  --open-handoff "${OPEN_HANDOFF}"
  --steer-k "${STEER_K}"
  --steer-chunks "${STEER_CHUNKS}"
  --stop-after "${STOP_AFTER}"
  --transition-window "${TRANSITION_WINDOW}"
)
if [[ "${LOAD_EARLY_STOP}" == "1" ]]; then
  PHASE1_ARGS+=(--load-early-stop)
fi
# --switch value scores handoffs with the same ensemble as BRIDGE=vsearch, which already
# passes these; adding them twice would only confuse the run tag's model field.
VALUE_ARGS=()
if [[ "${SWITCH_MODE}" == "value" ]]; then
  if [[ -z "${V_MODEL}" ]]; then
    echo "[CHAIN] SWITCH_MODE=value requires V_MODEL (checkpoints/.../model.pt)" >&2
    exit 2
  fi
  if [[ "${BRIDGE}" != "vsearch" ]]; then
    VALUE_ARGS+=(--v-model "${V_MODEL}" --v-threshold "${V_THRESHOLD}" --v-kappa "${V_KAPPA}")
  fi
fi
# Phase-3a/3b conditions: these change the run tag, so they go into both the
# --print-output-dir call and the run (the resume dir must match the run).
PERTURB_ARGS=()
if [[ -n "${PERTURB_DRAWER}" ]]; then
  read -r pdrawer_lo pdrawer_hi <<< "${PERTURB_DRAWER}"
  PERTURB_ARGS+=(--perturb-drawer "${pdrawer_lo}" "${pdrawer_hi}")
fi
if [[ -n "${OBJECT_DR}" ]]; then
  PERTURB_ARGS+=(--object-dr "${OBJECT_DR}")
fi
if [[ -n "${PERTURB_EE}" ]]; then
  PERTURB_ARGS+=(--perturb-ee "${PERTURB_EE}")
fi
if [[ -n "${DRAWER_LIMIT}" ]]; then
  PERTURB_ARGS+=(--drawer-limit "${DRAWER_LIMIT}")
fi
# Phase-5 per-handoff switching; these change the run tag, so they go into both calls.
case "${LOAD_SWITCH}" in flag|value) ;; *)
  echo "[CHAIN] LOAD_SWITCH must be flag|value (got '${LOAD_SWITCH}')" >&2; exit 2 ;; esac
case "${BRIDGE23}" in none|vsearch) ;; *)
  echo "[CHAIN] BRIDGE23 must be none|vsearch (got '${BRIDGE23}')" >&2; exit 2 ;; esac
PHASE5_ARGS=()
if [[ "${LOAD_SWITCH}" == "value" || "${BRIDGE23}" == "vsearch" ]]; then
  if [[ -z "${V3_MODEL}" ]]; then
    echo "[CHAIN] LOAD_SWITCH=value / BRIDGE23=vsearch require V3_MODEL" >&2
    exit 2
  fi
  PHASE5_ARGS+=(--v3-model "${V3_MODEL}")
fi
if [[ "${LOAD_SWITCH}" == "value" ]]; then PHASE5_ARGS+=(--load-switch value); fi
if [[ "${BRIDGE23}" == "vsearch" ]]; then PHASE5_ARGS+=(--bridge23 vsearch); fi
# Obstacle source; changes the run tag ("-pcd" for perceived), so both calls get it.
case "${OBSTACLES}" in privileged|perceived) ;; *)
  echo "[CHAIN] OBSTACLES must be privileged|perceived (got '${OBSTACLES}')" >&2; exit 2 ;; esac
OBSTACLE_ARGS=(--obstacles "${OBSTACLES}")
# V3 threshold/kappa override the V2 ones when set (they do not change the run tag).
if [[ -n "${V3_THRESHOLD:-}" ]]; then PHASE5_ARGS+=(--v3-threshold "${V3_THRESHOLD}"); fi
if [[ -n "${V3_KAPPA:-}" ]]; then PHASE5_ARGS+=(--v3-kappa "${V3_KAPPA}"); fi
# Value switch sanity floor ("" = the switcher's default 0.3). ee_est reads the floor off the
# scene estimator, which is ~0.15 high on a closed drawer, so a perception-only run needs it.
V_MIN_OPEN_FRAC="${V_MIN_OPEN_FRAC:-}"
if [[ -n "${V_MIN_OPEN_FRAC}" ]]; then PHASE5_ARGS+=(--v-min-open-frac "${V_MIN_OPEN_FRAC}"); fi
# Value feature sets (ee = privileged drawer frac; ee_est / img_ee = from the head image).
if [[ -n "${V_FEATURE_SET:-}" ]]; then PHASE5_ARGS+=(--v-feature-set "${V_FEATURE_SET}"); fi
if [[ -n "${V3_FEATURE_SET:-}" ]]; then PHASE5_ARGS+=(--v3-feature-set "${V3_FEATURE_SET}"); fi
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
# RTX 5090: OIDN CUDA unsupported → noisy svulkan2 errors / possible stall.
export SAPIEN_RT_DENOISER="${SAPIEN_RT_DENOISER:-none}"

# Fixed output dir, so a restart resumes from the same records.jsonl.
OUTPUT_DIR="$(python scripts/eval_cabinet_chain_naive.py --print-output-dir \
  --task-config "${TASK_CONFIG}" \
  --switch "${SWITCH_MODE}" \
  "${BRIDGE_ARGS[@]}" \
  --open-handoff "${OPEN_HANDOFF}" \
  --steer-k "${STEER_K}" \
  --transition-window "${TRANSITION_WINDOW}" \
  "${VALUE_ARGS[@]}" \
  "${PERTURB_ARGS[@]}" \
  "${PHASE5_ARGS[@]}" \
  "${OBSTACLE_ARGS[@]}")"
echo "[CHAIN] output dir: ${OUTPUT_DIR}"

RUN_ARGS=(
  scripts/eval_cabinet_chain_naive.py
  --task-config "${TASK_CONFIG}"
  --test-num "${TEST_NUM}"
  --seed-start "${SEED_START}"
  --policy-gpu "${POLICY_GPU}"
  --switch "${SWITCH_MODE}"
  "${BRIDGE_ARGS[@]}"
  --switch-every-n "${SWITCH_EVERY_N}"
  --min-steps-before-switch "${MIN_STEPS_BEFORE_SWITCH}"
  --vlm-model "${VLM_MODEL}"
  --vlm-conda-env "${VLM_CONDA_ENV}"
  --vlm-device "${VLM_DEVICE}"
  --video-substep "${VIDEO_SUBSTEP}"
  --stall-timeout-sec "${STALL_TIMEOUT_SEC}"
  --output-dir "${OUTPUT_DIR}"
  "${PHASE1_ARGS[@]}"
  "${VALUE_ARGS[@]}"
  "${PERTURB_ARGS[@]}"
  "${PHASE5_ARGS[@]}"
  "${OBSTACLE_ARGS[@]}"
)

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
    if [[ "${STALL_TIMEOUT_SEC}" -gt 0 && "${age}" -gt "${STALL_TIMEOUT_SEC}" ]]; then
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

echo "[CHAIN] eval finished (switch=${SWITCH_MODE}, exit=${code})"
exit "${code}"
