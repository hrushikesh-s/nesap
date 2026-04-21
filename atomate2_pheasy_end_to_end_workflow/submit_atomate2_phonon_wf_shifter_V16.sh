#!/bin/bash
# Atomate2-Pheasy phonon workflow launcher for Perlmutter
# V9: host srun -> shifter VASP launch model
#
# Key design change vs V8:
#   - DO NOT run a copied/staged srun inside the container.
#   - Run the jobflow/atomate2 Python driver on the host compute node.
#   - Launch VASP with native host srun, using shifter as the payload.
#
# This follows the debugging results that showed:
#   - native host srun works on the allocation
#   - copied srun inside Shifter kept failing on plugin/auth layers
#
# Usage:
#   1) Activate your host env first, e.g. conda activate atomate2_pheasy
#   2) bash submit_atomate2_phonon_wf_shifter_V9.sh

set -euo pipefail

CONTAINER_IMAGE="ghcr.io/hrushikesh-s/pheasy-phonopy:atomate2-pheasy-end2end-20260420"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_SCRIPT_DEFAULT="/global/homes/h/hrushi99/docker_build/run_phonon_workflow_V2.py"

clear
cat <<'BANNER'

╔══════════════════════════════════════════════════════════════╗
║    Atomate2-Pheasy End-to-End Phonon Workflow Launcher      ║
║    NERSC Perlmutter | Host srun → Shifter VASP  [V9]        ║
╚══════════════════════════════════════════════════════════════╝
BANNER

echo ""
echo "This version runs atomate2/jobflow on the host compute node and launches"
echo "VASP through native host srun with shifter as the payload."
echo ""

while true; do
    read -rp "[1/9] Compute type — CPU or GPU? (cpu/gpu): " COMPUTE_TYPE
    COMPUTE_TYPE="${COMPUTE_TYPE,,}"
    [[ "$COMPUTE_TYPE" == "cpu" || "$COMPUTE_TYPE" == "gpu" ]] && break
    echo "      ✗ Please enter 'cpu' or 'gpu'"
done

echo "      Note: NERSC interactive queue max = 4 nodes"
while true; do
    read -rp "[2/9] Number of nodes? (1–4): " N_NODES
    [[ "$N_NODES" =~ ^[1-4]$ ]] && break
    echo "      ✗ Please enter 1, 2, 3, or 4"
done

read -rp "[3/9] Materials Project ID? (e.g. mp-149): " MPID
MPID="${MPID:-mp-149}"

echo ""
echo "      Your API key: https://next-gen.materialsproject.org/api"
read -rp "[4/9] Materials Project API key: " MP_API_KEY
echo ""

DEFAULT_POTCAR="/pscratch/sd/h/hrushi99/POTCARs"
echo "      Default POTCAR path: ${DEFAULT_POTCAR}"
read -rp "[5/9] POTCAR directory path [Enter for default]: " POTCAR_DIR
POTCAR_DIR="${POTCAR_DIR:-$DEFAULT_POTCAR}"

echo ""
read -rp "[6/9] Output docs JSON path: " DOCS_STORE
DOCS_STORE="${DOCS_STORE:-/pscratch/sd/p/phillip/output_${MPID}.json}"

read -rp "[7/9] Output blob JSON path: " BLOB_STORE
BLOB_STORE="${BLOB_STORE:-/pscratch/sd/p/phillip/blob_${MPID}.json}"

read -rp "[8/9] SLURM account [default: matgen]: " SLURM_ACCOUNT
SLURM_ACCOUNT="${SLURM_ACCOUNT:-matgen}"

read -rp "[9/9] Workflow driver path [Enter for default]: " WORKFLOW_SCRIPT
WORKFLOW_SCRIPT="${WORKFLOW_SCRIPT:-$WORKFLOW_SCRIPT_DEFAULT}"

if [[ ! -f "$WORKFLOW_SCRIPT" ]]; then
    echo ""
    echo "ERROR: Workflow driver not found: $WORKFLOW_SCRIPT"
    exit 1
fi

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│                     SETTINGS SUMMARY                        │"
echo "├──────────────────────────────────────────────────────────────┤"
printf "│  %-22s %-39s│\n" "Compute type:"   "$COMPUTE_TYPE"
printf "│  %-22s %-39s│\n" "Nodes:"          "$N_NODES"
printf "│  %-22s %-39s│\n" "MP-ID:"          "$MPID"
printf "│  %-22s %-39s│\n" "MP-API key:"     "${MP_API_KEY:0:8}..."
printf "│  %-22s %-39s│\n" "POTCAR dir:"     "${POTCAR_DIR:0:39}"
printf "│  %-22s %-39s│\n" "Docs store:"     "${DOCS_STORE:0:39}"
printf "│  %-22s %-39s│\n" "Blob store:"     "${BLOB_STORE:0:39}"
printf "│  %-22s %-39s│\n" "SLURM account:"  "$SLURM_ACCOUNT"
printf "│  %-22s %-39s│\n" "Workflow script:" "$(basename "$WORKFLOW_SCRIPT")"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

while true; do
    read -rp ">>> Proceed with these settings? (y/n): " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] && break
    [[ "$CONFIRM" =~ ^[Nn]$ ]] && echo "Aborted." && exit 0
done

if [[ "$COMPUTE_TYPE" == "gpu" ]]; then
    CONSTRAINT="gpu"
    N_TASKS=$((N_NODES * 4))
    SRUN_FLAGS_STD="-N${N_NODES} -n${N_TASKS} -c32 --cpu-bind=cores -G${N_TASKS} --gpu-bind=none"
    SRUN_FLAGS_GAM="-N${N_NODES} -n${N_TASKS} -c32 --cpu-bind=cores -G${N_TASKS} --gpu-bind=none"
    VASP_MODULE="vasp/6.4.2-gpu"
    OMP_NUM_THREADS=1
    OMP_PLACES="threads"
    SHIFTER_MODULE="--module=gpu,mpich"
else
    CONSTRAINT="cpu"
    N_TASKS=$((N_NODES * 16))
    SRUN_FLAGS_STD="--exclusive -N${N_NODES} -n${N_TASKS} -c16 --cpu-bind=cores"
    SRUN_FLAGS_GAM="--exclusive -N${N_NODES} -n${N_TASKS} -c16 --cpu-bind=cores"
    VASP_MODULE="vasp/6.4.2-cpu"
    OMP_NUM_THREADS=8
    OMP_PLACES="cores"
    SHIFTER_MODULE="--module=mpich"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR="${SCRIPT_DIR}/configs_${MPID}_${TIMESTAMP}"
LOG_DIR="${SCRIPT_DIR}/logs_${MPID}_${TIMESTAMP}"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/master_${MPID}_${TIMESTAMP}.log"
ATOMATE2_CFG="${CONFIG_DIR}/atomate2.yaml"
JOBFLOW_CFG="${CONFIG_DIR}/jobflow.yaml"
PMGRC_CFG="${CONFIG_DIR}/.pmgrc.yaml"
PY_BOOTSTRAP="${CONFIG_DIR}/run_phonon_workflow_host_srun.py"
INNER_SCRIPT="/tmp/atomate2_inner_${MPID}_$$.sh"

cat > "$JOBFLOW_CFG" <<EOF2
## Atomate2-Pheasy — Auto-generated: $(date)
JOB_STORE:
  docs_store:
    type: JSONStore
    paths: ${DOCS_STORE}
    read_only: False
  additional_stores:
    data:
      type: JSONStore
      paths: ${BLOB_STORE}
      read_only: False
EOF2

cat > "$PMGRC_CFG" <<EOF2
PMG_VASP_PSP_DIR: ${POTCAR_DIR}
EOF2













cat > "$PY_BOOTSTRAP" <<EOF2
#!/usr/bin/env python3
import importlib.util
import pathlib
import sys

workflow_path = pathlib.Path(r"${WORKFLOW_SCRIPT}")
if not workflow_path.is_file():
    raise SystemExit(f"Workflow driver not found: {workflow_path}")

spec = importlib.util.spec_from_file_location("run_phonon_workflow_V2_host", workflow_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not import workflow driver from {workflow_path}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def _skip_preflight():
    print("[V9] Skipping V8 staged-srun preflight; using host srun -> shifter VASP launch model", flush=True)

mod.preflight_slurm_check = _skip_preflight

if __name__ == "__main__":
    mod.main()
EOF2
chmod +x "$PY_BOOTSTRAP"

echo ""
echo "[$(date '+%H:%M:%S')] Config/helper files written."
echo "[$(date '+%H:%M:%S')] Logs: ${LOG_DIR}"

cat > "$INNER_SCRIPT" <<EOF2
#!/bin/bash
set -euo pipefail

source /etc/profile.d/z00_lmod.sh 2>/dev/null || \
source /usr/share/lmod/lmod/init/bash 2>/dev/null || true

MASTER_LOG="${MASTER_LOG}"
_log() { echo "\$1" | tee -a "\${MASTER_LOG}"; }

_log ""
_log "════════════════════════════════════════════════════════════════"
_log "  WORKFLOW STARTING — ${MPID} — ${COMPUTE_TYPE} x ${N_NODES} nodes"
_log "  Started  : \$(date '+%Y-%m-%d %H:%M:%S')"
_log "  Nodes    : \${SLURM_JOB_NODELIST:-unknown}"
_log "  Model    : host srun → shifter VASP"
_log "════════════════════════════════════════════════════════════════"

WF_START=\$(date +%s)

_log "[V9] Loading VASP module: ${VASP_MODULE}"
module load ${VASP_MODULE}

SRUN_HOST=\$(which srun 2>/dev/null || true)
VASP_STD_BIN=\$(which vasp_std 2>/dev/null || true)
VASP_GAM_BIN=\$(which vasp_gam 2>/dev/null || true)
PYTHON_BIN=\$(which python 2>/dev/null || true)

[[ -z "\${SRUN_HOST}" ]] && { _log "ERROR: srun not found."; exit 1; }
[[ -z "\${PYTHON_BIN}" ]] && { _log "ERROR: python not found."; exit 1; }
[[ -z "\${VASP_STD_BIN}" ]] && { _log "ERROR: vasp_std not found after module load."; exit 1; }
[[ -z "\${VASP_GAM_BIN}" ]] && VASP_GAM_BIN="\$(dirname "\${VASP_STD_BIN}")/vasp_gam"

_log "[V9] srun       : \${SRUN_HOST}"
_log "[V9] python     : \${PYTHON_BIN}"
_log "[V9] vasp_std   : \${VASP_STD_BIN}"
_log "[V9] vasp_gam   : \${VASP_GAM_BIN}"
_log "[V9] container  : ${CONTAINER_IMAGE}"




# No Shifter wrapper for VASP. Use host VASP directly.
cat > "${ATOMATE2_CFG}" <<ATOMATE2EOF
# Auto-generated on \$(hostname) — \$(date)
# V9: host srun launches host VASP directly.
VASP_CMD: \${SRUN_HOST} ${SRUN_FLAGS_STD} \${VASP_STD_BIN}
VASP_GAMMA_CMD: \${SRUN_HOST} ${SRUN_FLAGS_GAM} \${VASP_GAM_BIN}
ATOMATE2EOF











_log "[V9] atomate2.yaml → ${ATOMATE2_CFG}"
_log "[V9] Host-side VASP launch model configured."

export JOBFLOW_CONFIG_FILE="${JOBFLOW_CFG}"
export ATOMATE2_CONFIG_FILE="${ATOMATE2_CFG}"
export PMG_VASP_PSP_DIR="${POTCAR_DIR}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export OMP_PLACES=${OMP_PLACES}
export OMP_PROC_BIND=spread
export FI_CXI_RX_MATCH_MODE=hybrid

_log "[V9] Preparing clean host python package overlay..."

export PYTHONNOUSERSITE=1










export HOST_PY_PKGS="/pscratch/sd/h/hrushi99/host_py_pkgs"
mkdir -p "\${HOST_PY_PKGS}"


rm -rf "\${HOST_PY_PKGS}/emmet" \
       "\${HOST_PY_PKGS}/mp_api" \
       "\${HOST_PY_PKGS}/pymatgen" \
       "\${HOST_PY_PKGS}"/emmet_core-*.dist-info \
       "\${HOST_PY_PKGS}"/mp_api-*.dist-info \
       "\${HOST_PY_PKGS}"/pymatgen-*.dist-info \
       "\${HOST_PY_PKGS}/bin"


"\${PYTHON_BIN}" -m pip install -q --upgrade --force-reinstall --no-cache-dir --no-deps \
    --target="\${HOST_PY_PKGS}" \
    "emmet-core==0.84.2" \
    "mp-api==0.41.2" \
    "pymatgen==2024.8.9"







export PYTHONPATH="\${HOST_PY_PKGS}:\${PYTHONPATH:-}"

_log "[V9] Verifying host python environment imports..."
PYTHONNOUSERSITE=1 PYTHONPATH="\${HOST_PY_PKGS}:\${PYTHONPATH:-}" "\${PYTHON_BIN}" - <<'PYCHECK'
import emmet, mp_api, pymatgen
print("emmet   :", emmet.__file__)
print("mp_api  :", mp_api.__file__)
print("pymatgen:", pymatgen.__file__)

from mp_api.client import MPRester
from emmet.core.phonon import PhononBSDOSDoc

print("[V9] Host python env check passed")
PYCHECK

_log "[V9] Launching workflow driver on host..."






PYTHONNOUSERSITE=1 PYTHONPATH="\${HOST_PY_PKGS}:\${PYTHONPATH:-}" \
"\${PYTHON_BIN}" "${PY_BOOTSTRAP}" \
    --mpid "${MPID}" \
    --mp-api-key "${MP_API_KEY}" \
    --docs-store "${DOCS_STORE}" \
    --blob-store "${BLOB_STORE}" \
    --logdir "${LOG_DIR}" \
    2> >(tee -a "${LOG_DIR}/stderr_${MPID}_${TIMESTAMP}.err" >&2) \
    | tee -a "\${MASTER_LOG}"



WF_EXIT=\${PIPESTATUS[0]}
WF_END=\$(date +%s)
TOTAL_SEC=\$((WF_END - WF_START))
TOTAL_H=\$((TOTAL_SEC / 3600))
















TOTAL_M=\$(((TOTAL_SEC % 3600) / 60))
TOTAL_S=\$((TOTAL_SEC % 60))

_log ""
_log "════════════════════════════════════════════════════════════════"
_log "  \$( [ \$WF_EXIT -eq 0 ] && echo '✔ WORKFLOW COMPLETE' || echo '✗ WORKFLOW FAILED (exit: '\${WF_EXIT}')' )"
_log "  MP-ID : ${MPID}   Finished : \$(date '+%Y-%m-%d %H:%M:%S')"
_log "  Time  : \${TOTAL_H}h \${TOTAL_M}m \${TOTAL_S}s    Log: \${MASTER_LOG}"
_log "════════════════════════════════════════════════════════════════"

exit \$WF_EXIT
EOF2
chmod +x "$INNER_SCRIPT"

echo ""
echo "[$(date '+%H:%M:%S')] Requesting interactive allocation..."
echo "  → salloc -N ${N_NODES} -C ${CONSTRAINT} -A ${SLURM_ACCOUNT} -t 4:00:00 --qos=interactive"
echo ""

salloc \
    -N "$N_NODES" \
    -C "$CONSTRAINT" \
    -A "$SLURM_ACCOUNT" \
    -t 4:00:00 \
    --qos=interactive \
    bash "$INNER_SCRIPT"

SALLOC_EXIT=$?
rm -f "$INNER_SCRIPT"

echo ""
if [[ $SALLOC_EXIT -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')] ✔ All done."
else
    echo "[$(date '+%H:%M:%S')] ✗ Exit code ${SALLOC_EXIT}."
fi
echo "  Logs    : ${LOG_DIR}"
echo "  Configs : ${CONFIG_DIR}"
