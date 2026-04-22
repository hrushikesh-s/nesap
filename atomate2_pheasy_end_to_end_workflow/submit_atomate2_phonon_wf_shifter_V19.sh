#!/bin/bash
# Atomate2-Pheasy phonon workflow launcher for NERSC Perlmutter
# V19: FIXED heredoc escaping bug present in V18.
#      + FIXED segfault root causes (see patch notes below).
#
# ROOT CAUSE of V18 failure
# ─────────────────────────
#   The inner script was generated with an UNQUOTED heredoc (<<EOF2).
#   In bash, an unquoted heredoc:
#     • Expands ${VAR} immediately on the login node at write-time, so
#       variables like ${LD_LIBRARY_PATH_INNER} (defined only at compute-
#       node runtime) expanded to empty string in the generated file.
#     • Treats a trailing backslash as a line-continuation, collapsing
#         source ... || \<newline>source ...
#       into one physical line — shifting all subsequent line numbers so
#       that WF_START=\$(date +%s) arrived in the file with the backslash
#       still intact, producing the "syntax error near unexpected token '('"
#       at line 18.
#     • Ran $(scontrol show config ...) on the login node rather than the
#       compute node, producing wrong (or empty) PluginDir values.
#
# FIX (V19)
# ─────────
#   • Inner script is written with a SINGLE-QUOTED heredoc (<<'INNER_EOF')
#     so every byte is written verbatim — no expansion, no line-continuation.
#   • All outer-script variables needed by the inner script are EXPORTED
#     before calling salloc.  SLURM's default --export=ALL passes the full
#     caller environment into the job, so those variables are live when the
#     inner script runs on the compute node.
#   • SLURM_* variables are NOT stripped (no --clearenv) so the staged srun
#     can authenticate with the SLURM daemon using the job's own credentials.
#   • atomate2.yaml is written inside the inner script (after module load
#     resolves the real vasp_std/vasp_gam paths) using a normal heredoc that
#     executes at compute-node runtime — exactly as intended.
#
# SEGFAULT FIX (V19 patch)
# ────────────────────────
#   Three root causes were diagnosed from the std_err.txt segfaults and OUTCAR:
#
#   1. NPAR=12 hard-coded in run_phonon_workflow_V4.py
#      VASP requires KPAR × NPAR × NCORE = N_MPI_ranks.  With NCORE=1 and
#      N_MPI=16 (4 nodes × 4 GPUs), NPAR=12 gives KPAR=16/12=1.33 → not
#      an integer.  VASP auto-adjusts to 16 groups (visible in OUTCAR:
#      "one band on NCORE= 1 cores, 16 groups"), but then segfaults during
#      band distribution for small-NBANDS structures like Si (mp-149).
#      FIX: export SLURM_NNODES into the container so the Python script can
#      read it and set NPAR = SLURM_NNODES (4 for 4 nodes → 16/4=4 ✔).
#      The Python script already reads SLURM_NNODES; SLURM forwards it
#      automatically but we also --env it explicitly for safety.
#
#   2. Cray-MPICH GPU Transport Layer (GTL) missing from HOST_LD_LIBRARY_PATH
#      "module load vasp/6.4.2-gpu" sets CRAY_LD_LIBRARY_PATH (a SEPARATE
#      variable from LD_LIBRARY_PATH) which contains libmpi_gtl_cuda.so.
#      The old code only captured LD_LIBRARY_PATH, so the GTL path was silently
#      dropped.  Without libmpi_gtl_cuda.so, all GPU-direct MPI calls
#      (MPI_Allreduce, MPI_Bcast on device buffers) segfault after OpenACC init.
#      FIX: explicitly prepend CRAY_LD_LIBRARY_PATH and the gtl/lib directory
#      found under /opt/cray/pe/mpich/*/gtl/lib to HOST_LD_LIBRARY_PATH.
#
#   3. MPICH_GPU_SUPPORT_ENABLED=1 not passed to VASP tasks
#      Cray-MPICH needs this env var to activate the GPU transport layer.
#      Without it, the GTL is loaded but MPI ignores it for GPU buffers.
#      FIX: add to SHIFTER_ARGS (→ container env) and to srun --export
#      (→ VASP task env).
#
# Execution model
# ───────────────
#   Run the atomate2/jobflow Python workflow driver inside the Shifter image.
#   VASP, srun, and SLURM remain on the NERSC host.
#   The host srun binary and its shared libraries are staged into a scratch-
#   backed directory that is visible inside the container, per the pattern
#   described in https://docs.nersc.gov/development/containers/shifter/how-to-use/
#
# Usage
# ─────
#   1) shifterimg pull ghcr.io/hrushikesh-s/pheasy-phonopy:atomate2-pheasy-end2end-20260420
#   2) bash submit_atomate2_phonon_wf_shifter_V19.sh

# ─── Banner ───────────────────────────────────────────────────────────────────

clear
cat <<'BANNER'

╔════════════════════════════════════════════════════════════════════╗
║      Atomate2-Pheasy End-to-End Phonon Workflow Launcher         ║
║   NERSC Perlmutter | Shifter driver + host NERSC VASP  [V19]     ║
╚════════════════════════════════════════════════════════════════════╝
BANNER

echo ""
echo "This version runs the atomate2/jobflow workflow driver inside the"
echo "Shifter image, while VASP, srun, and SLURM remain on the NERSC host."
echo ""

# ─── Constants ────────────────────────────────────────────────────────────────

CONTAINER_IMAGE="ghcr.io/hrushikesh-s/pheasy-phonopy:atomate2-pheasy-end2end-20260420"
WORKFLOW_SCRIPT_DEFAULT="/global/homes/h/hrushi99/docker_build/run_phonon_workflow_V4.py"

# ─── Interactive prompts ──────────────────────────────────────────────────────

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
DOCS_STORE="${DOCS_STORE:-/global/homes/h/hrushi99/docker_build/docs_${MPID}.json}"

read -rp "[7/9] Output blob JSON path: " BLOB_STORE
BLOB_STORE="${BLOB_STORE:-/global/homes/h/hrushi99/docker_build/blob_${MPID}.json}"

read -rp "[8/9] SLURM account [default: matgen]: " SLURM_ACCOUNT
SLURM_ACCOUNT="${SLURM_ACCOUNT:-matgen}"

read -rp "[9/9] Workflow driver path [Enter for default]: " WORKFLOW_SCRIPT
WORKFLOW_SCRIPT="${WORKFLOW_SCRIPT:-$WORKFLOW_SCRIPT_DEFAULT}"

if [[ ! -f "$WORKFLOW_SCRIPT" ]]; then
    echo ""
    echo "ERROR: Workflow driver not found: $WORKFLOW_SCRIPT"
    exit 1
fi

# ─── Settings summary ─────────────────────────────────────────────────────────

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│                     SETTINGS SUMMARY                        │"
echo "├──────────────────────────────────────────────────────────────┤"
printf "│  %-22s %-39s│\n" "Compute type:"    "$COMPUTE_TYPE"
printf "│  %-22s %-39s│\n" "Nodes:"           "$N_NODES"
printf "│  %-22s %-39s│\n" "MP-ID:"           "$MPID"
printf "│  %-22s %-39s│\n" "MP-API key:"      "${MP_API_KEY:0:8}..."
printf "│  %-22s %-39s│\n" "POTCAR dir:"      "${POTCAR_DIR:0:39}"
printf "│  %-22s %-39s│\n" "Docs store:"      "${DOCS_STORE:0:39}"
printf "│  %-22s %-39s│\n" "Blob store:"      "${BLOB_STORE:0:39}"
printf "│  %-22s %-39s│\n" "SLURM account:"   "$SLURM_ACCOUNT"
printf "│  %-22s %-39s│\n" "Workflow script:" "$(basename "$WORKFLOW_SCRIPT")"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

while true; do
    read -rp "Proceed with these settings? (y/n): " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] && break
    [[ "$CONFIRM" =~ ^[Nn]$ ]] && echo "Aborted." && exit 0
done

# ─── Compute-type settings ────────────────────────────────────────────────────

if [[ "$COMPUTE_TYPE" == "gpu" ]]; then
    CONSTRAINT="gpu"
    N_TASKS=$((N_NODES * 4))          # 4 GPUs per node on Perlmutter
    SRUN_FLAGS_STD="-N${N_NODES} -n${N_TASKS} -c32 --cpu-bind=cores -G${N_TASKS} --gpu-bind=none"
    SRUN_FLAGS_GAM="-N${N_NODES} -n${N_TASKS} -c32 --cpu-bind=cores -G${N_TASKS} --gpu-bind=none"
    VASP_MODULE="vasp/6.4.2-gpu"
    OMP_NUM_THREADS=1
    OMP_PLACES="threads"
    SHIFTER_MODULE="--module=gpu,mpich"
    MPICH_GPU_SUPPORT=1   # ← NEW: needed for Cray-MPICH GPU transport layer
else
    CONSTRAINT="cpu"
    N_TASKS=$((N_NODES * 16))
    SRUN_FLAGS_STD="--exclusive -N${N_NODES} -n${N_TASKS} -c16 --cpu-bind=cores"
    SRUN_FLAGS_GAM="--exclusive -N${N_NODES} -n${N_TASKS} -c16 --cpu-bind=cores"
    VASP_MODULE="vasp/6.4.2-cpu"
    OMP_NUM_THREADS=8
    OMP_PLACES="cores"
    SHIFTER_MODULE="--module=mpich"
    MPICH_GPU_SUPPORT=0
fi

# ─── Directories + timestamps ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_DIR="${SCRIPT_DIR}/configs_${MPID}_${TIMESTAMP}"
LOG_DIR="${SCRIPT_DIR}/logs_${MPID}_${TIMESTAMP}"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/master_${MPID}_${TIMESTAMP}.log"
ATOMATE2_CFG="${CONFIG_DIR}/atomate2.yaml"
JOBFLOW_CFG="${CONFIG_DIR}/jobflow.yaml"
PMGRC_CFG="${CONFIG_DIR}/.pmgrc.yaml"
PY_BOOTSTRAP="${CONFIG_DIR}/run_phonon_workflow_bootstrap.py"
INNER_SCRIPT="/tmp/atomate2_inner_${MPID}_$$.sh"
NERSC_RUNTIME_DIR="${CONFIG_DIR}/nersc_runtime"
CONTAINER_PY_PKGS="${CONFIG_DIR}/container_py_pkgs"

# ─── Write jobflow.yaml ───────────────────────────────────────────────────────
# Written on the login node — paths are known here.

cat > "$JOBFLOW_CFG" << JFEOF
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
JFEOF

# ─── Write .pmgrc.yaml ────────────────────────────────────────────────────────

cat > "$PMGRC_CFG" << PMGEOF
PMG_VASP_PSP_DIR: ${POTCAR_DIR}
PMGEOF

# ─── Write Python bootstrap shim ─────────────────────────────────────────────
# This shim is called by shifter inside the container.  It loads the workflow
# driver from the shared filesystem (visible to Shifter) and calls main().

cat > "$PY_BOOTSTRAP" << PYEOF
#!/usr/bin/env python3
import importlib.util, pathlib, sys

workflow_path = pathlib.Path("${WORKFLOW_SCRIPT}")
if not workflow_path.is_file():
    raise SystemExit(f"Workflow driver not found: {workflow_path}")

spec = importlib.util.spec_from_file_location("run_phonon_workflow_V4", workflow_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not load spec from {workflow_path}")

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.main()
PYEOF
chmod +x "$PY_BOOTSTRAP"

echo ""
echo "[$(date '+%H:%M:%S')] Config/helper files written."
echo "[$(date '+%H:%M:%S')] Logs: ${LOG_DIR}"

# ─── Export all variables the inner script needs ──────────────────────────────
# salloc forwards the caller's full environment into the job (default
# --export=ALL), so every exported variable is available when the inner
# script runs on the compute node — no need to bake values into the file.

export CONTAINER_IMAGE MPID MP_API_KEY POTCAR_DIR DOCS_STORE BLOB_STORE
export LOG_DIR CONFIG_DIR TIMESTAMP MASTER_LOG
export ATOMATE2_CFG JOBFLOW_CFG PMGRC_CFG PY_BOOTSTRAP
export NERSC_RUNTIME_DIR CONTAINER_PY_PKGS
export VASP_MODULE N_NODES N_TASKS SRUN_FLAGS_STD SRUN_FLAGS_GAM
export OMP_NUM_THREADS OMP_PLACES SHIFTER_MODULE COMPUTE_TYPE
export MPICH_GPU_SUPPORT   # ← NEW: 1 for GPU, 0 for CPU

# ─── Write inner script ───────────────────────────────────────────────────────
# CRITICAL: the heredoc delimiter is SINGLE-QUOTED (<<'INNER_EOF').
# This means the entire body is written VERBATIM — no variable expansion,
# no command substitution, no backslash processing occurs at write time.
# Every $VAR, $(cmd), and \n in the body is literal text in the output file.
# Variable expansion and command substitution happen later when bash executes
# the inner script on the compute node, which is exactly what we want.

cat > "$INNER_SCRIPT" <<'INNER_EOF'
#!/bin/bash
# Inner script — runs on the allocated compute node(s).
# All configuration variables arrive via the inherited SLURM environment
# (exported by the outer script before salloc).
#set -euo pipefail

# ── Enable module system ──────────────────────────────────────────────────────
# Lmod is not sourced automatically on compute nodes in all contexts.
source /etc/profile.d/z00_lmod.sh   2>/dev/null \
    || source /usr/share/lmod/lmod/init/bash 2>/dev/null \
    || true

# ── Logging helper ────────────────────────────────────────────────────────────
_log() { echo "$1" | tee -a "${MASTER_LOG}"; }

_log ""
_log "════════════════════════════════════════════════════════════════"
_log "  WORKFLOW STARTING — ${MPID} — ${COMPUTE_TYPE} x ${N_NODES} nodes"
_log "  Started  : $(date '+%Y-%m-%d %H:%M:%S')"
_log "  Nodes    : ${SLURM_JOB_NODELIST:-unknown}"
_log "  Job ID   : ${SLURM_JOB_ID:-unknown}"
_log "  Model    : Shifter Python driver + host VASP/srun (V19)"
_log "════════════════════════════════════════════════════════════════"

WF_START=$(date +%s)

# ── Load host VASP module ─────────────────────────────────────────────────────
_log "[V19] Loading host VASP module: ${VASP_MODULE}"
module load "${VASP_MODULE}"

# ── Capture HOST_LD_LIBRARY_PATH ─────────────────────────────────────────────
# Capture the full LD_LIBRARY_PATH set by the VASP module.  This includes
# ROCm/CUDA, Intel MKL, Cray libfabric, etc.
HOST_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# ── FIX: Prepend Cray-MPICH GTL (GPU Transport Layer) paths ──────────────────
#
# ROOT CAUSE OF SEGFAULT #2:
# "module load vasp/6.4.2-gpu" sets CRAY_LD_LIBRARY_PATH (a SEPARATE variable
# from LD_LIBRARY_PATH) which contains libmpi_gtl_cuda.so.  Without this
# library, all GPU-direct MPI operations (device-buffer Allreduce/Bcast) segfault
# immediately after OpenACC initializes.
#
# We also explicitly locate the Cray-MPICH gtl/lib directory in case
# CRAY_LD_LIBRARY_PATH is not set in this execution context.
#
if [[ "${COMPUTE_TYPE}" == "gpu" ]]; then
    # 1. Use CRAY_LD_LIBRARY_PATH if the module set it
    if [[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]]; then
        HOST_LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${HOST_LD_LIBRARY_PATH}"
        _log "[V19] CRAY_LD_LIBRARY_PATH prepended to HOST_LD_LIBRARY_PATH"
    fi

    # 2. Also find the GTL lib dir explicitly (belt-and-suspenders)
    #    Typical location: /opt/cray/pe/mpich/<ver>/gtl/lib
    _CRAY_GTL=$(ls -d /opt/cray/pe/mpich/*/gtl/lib 2>/dev/null | sort -V | tail -1)
    if [[ -n "${_CRAY_GTL}" && -d "${_CRAY_GTL}" ]]; then
        # Only prepend if not already present
        case ":${HOST_LD_LIBRARY_PATH}:" in
            *":${_CRAY_GTL}:"*) ;;
            *) HOST_LD_LIBRARY_PATH="${_CRAY_GTL}:${HOST_LD_LIBRARY_PATH}"
               _log "[V19] Cray GTL path prepended : ${_CRAY_GTL}" ;;
        esac
    else
        _log "[V19] WARNING: Cray GTL lib dir not found under /opt/cray/pe/mpich/*/gtl/lib"
        _log "[V19]          Check 'module show ${VASP_MODULE}' for the correct path."
    fi

    # 3. Verify libmpi_gtl_cuda.so is now findable
    if ldconfig -p 2>/dev/null | grep -q 'libmpi_gtl_cuda' \
       || find ${_CRAY_GTL:-/opt/cray} -name 'libmpi_gtl_cuda.so*' -maxdepth 4 \
              2>/dev/null | grep -q .; then
        _log "[V19] libmpi_gtl_cuda.so : found ✔"
    else
        _log "[V19] WARNING: libmpi_gtl_cuda.so not located — GPU-direct MPI may fail"
    fi
fi

SRUN_HOST=$(which srun     2>/dev/null || true)
SHIFTER_BIN=$(which shifter  2>/dev/null || true)
VASP_STD_BIN=$(which vasp_std  2>/dev/null || true)
VASP_GAM_BIN=$(which vasp_gam  2>/dev/null || true)

if [[ -z "${SRUN_HOST}" ]]; then
    _log "ERROR: srun not found on PATH after module load."
    exit 1
fi
if [[ -z "${SHIFTER_BIN}" ]]; then
    _log "ERROR: shifter not found on PATH."
    exit 1
fi
if [[ -z "${VASP_STD_BIN}" ]]; then
    _log "ERROR: vasp_std not found on PATH after loading ${VASP_MODULE}."
    exit 1
fi
if [[ -z "${VASP_GAM_BIN}" ]]; then
    VASP_GAM_BIN="$(dirname "${VASP_STD_BIN}")/vasp_gam"
    _log "[V19] vasp_gam not on PATH; assuming: ${VASP_GAM_BIN}"
fi

_log "[V19] srun      : ${SRUN_HOST}"
_log "[V19] shifter   : ${SHIFTER_BIN}"
_log "[V19] vasp_std  : ${VASP_STD_BIN}"
_log "[V19] vasp_gam  : ${VASP_GAM_BIN}"
_log "[V19] container : ${CONTAINER_IMAGE}"

# ── Stage host srun + shared libraries + SLURM plugins ───────────────────────
#
# NERSC Shifter images do not bundle the host SLURM runtime.  We stage the
# host srun binary, its ELF-dynamic dependencies (via ldd), and the SLURM
# plugin .so files into a scratch-backed directory tree, then expose that
# tree inside the container via --volume (or by placing it on a shared FS
# already visible inside Shifter).  A patched slurm.conf whose PluginDir
# points to the staged plugins completes the environment so srun can
# authenticate with the SLURM daemon and launch host-side VASP processes.
#
# Reference: https://docs.nersc.gov/development/containers/shifter/how-to-use/

STAGED_BIN_DIR="${NERSC_RUNTIME_DIR}/bin"
STAGED_ETC_DIR="${NERSC_RUNTIME_DIR}/etc"
STAGED_PLUGIN_DIR="${NERSC_RUNTIME_DIR}/lib64/slurm"
STAGED_LIB_DIR="${NERSC_RUNTIME_DIR}/lib64"
STAGED_SRUN="${STAGED_BIN_DIR}/srun"
STAGED_SLURM_CONF="${STAGED_ETC_DIR}/slurm.conf"

mkdir -p "${STAGED_BIN_DIR}" "${STAGED_ETC_DIR}" \
         "${STAGED_PLUGIN_DIR}" "${STAGED_LIB_DIR}" \
         "${CONTAINER_PY_PKGS}"

# ── Locate host slurm.conf ────────────────────────────────────────────────────
HOST_SLURM_CONF="${SLURM_CONF:-}"
if [[ -z "${HOST_SLURM_CONF}" || ! -f "${HOST_SLURM_CONF}" ]]; then
    for _cand in /run/slurm/conf/slurm.conf \
                 /etc/slurm/slurm.conf       \
                 /etc/slurm.conf             \
                 /usr/etc/slurm.conf; do
        if [[ -f "${_cand}" ]]; then
            HOST_SLURM_CONF="${_cand}"
            break
        fi
    done
fi
if [[ -z "${HOST_SLURM_CONF}" || ! -f "${HOST_SLURM_CONF}" ]]; then
    _log "ERROR: cannot locate host slurm.conf"
    exit 1
fi

# ── Locate host SLURM PluginDir ───────────────────────────────────────────────
# Run scontrol on the compute node (this is inside the inner script, so
# it executes at runtime — not on the login node as it did in V18).
HOST_PLUGIN_DIR="$(scontrol show config 2>/dev/null \
    | awk -F= '/^PluginDir[[:space:]]*=/{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2; exit}')"
if [[ -z "${HOST_PLUGIN_DIR}" ]]; then
    HOST_PLUGIN_DIR="$(awk -F= '/^PluginDir=/{print $2; exit}' \
        "${HOST_SLURM_CONF}" | xargs 2>/dev/null || true)"
fi
if [[ -z "${HOST_PLUGIN_DIR}" || ! -d "${HOST_PLUGIN_DIR}" ]]; then
    _log "ERROR: cannot locate host SLURM PluginDir"
    exit 1
fi

# ── Stage srun binary ─────────────────────────────────────────────────────────
cp -Lf "${SRUN_HOST}" "${STAGED_SRUN}"
chmod +x "${STAGED_SRUN}"

# ── Stage slurm.conf with PluginDir redirected to staged location ─────────────
cp -f "${HOST_SLURM_CONF}" "${STAGED_SLURM_CONF}"
if grep -q '^PluginDir=' "${STAGED_SLURM_CONF}"; then
    sed -i "s|^PluginDir=.*|PluginDir=${STAGED_PLUGIN_DIR}|" "${STAGED_SLURM_CONF}"
else
    printf '\nPluginDir=%s\n' "${STAGED_PLUGIN_DIR}" >> "${STAGED_SLURM_CONF}"
fi
# Remove SlurmUser from the staged config.  Inside a Shifter container the
# process runs as the real user's UID (not root), so srun emits
# "Invalid user for SlurmUser root, ignored / Unable to process configuration
# file" and refuses to start.  The srun CLIENT has no need for this field —
# it is a server-side (slurmctld) setting — so removing it is safe.
sed -i '/^SlurmUser=/d; /^CliFilterPlugins=/d' "${STAGED_SLURM_CONF}"

# ── Stage SLURM plugins ───────────────────────────────────────────────────────
find "${STAGED_PLUGIN_DIR}" -maxdepth 1 -type f -delete 2>/dev/null || true
cp -Lf "${HOST_PLUGIN_DIR}"/*.so "${STAGED_PLUGIN_DIR}/" 2>/dev/null || true

# ── Stage srun's ELF-dynamic library dependencies (via ldd) ──────────────────
# IMPORTANT: we EXCLUDE standard glibc / kernel-interface libraries
# (libc, libpthread, libm, libdl, librt, libgcc_s, linux-vdso, ...).
# The Shifter container already ships its own versions of those.  Copying the
# HOST copies and prepending them via LD_LIBRARY_PATH causes a glibc ABI
# conflict ("undefined symbol: __nptl_change_stack_perm, version GLIBC_PRIVATE").
# We only stage SLURM-specific libraries (libslurm, libpmi*, libmunge, etc.)
# that the container genuinely does not have.
#
# We run ldd on BOTH the srun binary AND every staged plugin .so, because
# plugins have their own dependencies (e.g., cli_filter_lua.so needs
# liblua5.3.so.5) that are absent from the container and must be staged.
_GLIBC_FILTER='(^linux-vdso|/libc\.so\.|/libpthread\.so\.|/libm\.so\.|/libdl\.so\.|/librt\.so\.|/libgcc_s\.so\.|/libcrypt\.so\.|/libresolv\.so\.|/libutil\.so\.)'

find "${STAGED_LIB_DIR}" -maxdepth 1 -type f -delete 2>/dev/null || true

# Pass 1 — srun binary itself
ldd "${SRUN_HOST}" 2>/dev/null \
    | awk '$2=="=>" && $3~/^\// {print $3}; $1~/^\// {print $1}' \
    | sort -u \
    | grep -Ev "${_GLIBC_FILTER}" \
    | while IFS= read -r _dep; do
        [[ -f "${_dep}" ]] && cp -Lf "${_dep}" "${STAGED_LIB_DIR}/" 2>/dev/null || true
    done

# Pass 2 — every staged plugin .so (catches liblua5.3, libhttp_parser, etc.)
find "${STAGED_PLUGIN_DIR}" -maxdepth 1 -name '*.so' \
    | while IFS= read -r _plugin; do
        ldd "${_plugin}" 2>/dev/null \
            | awk '$2=="=>" && $3~/^\// {print $3}; $1~/^\// {print $1}'
    done \
    | sort -u \
    | grep -Ev "${_GLIBC_FILTER}" \
    | while IFS= read -r _dep; do
        [[ -f "${_dep}" ]] && cp -Lf "${_dep}" "${STAGED_LIB_DIR}/" 2>/dev/null || true
    done

# Combined LD_LIBRARY_PATH for the container environment (for the srun PROCESS).
# Layout (left = highest priority):
#   1. STAGED_LIB_DIR   — SLURM-specific libs (libslurm, libpmi, liblua, …)
#   2. STAGED_PLUGIN_DIR — SLURM plugins
#   3. HOST_LD_LIBRARY_PATH — everything the VASP module added (ROCm, Cray-MPICH
#                             GPU transport libmpi_gtl_cuda.so, MKL, CUDA, etc.)
# Without (3), srun-spawned VASP tasks inherit a stripped LD_LIBRARY_PATH that
# lacks the GPU/MPI libraries and segfault immediately.
LD_LIBRARY_PATH_INNER="${STAGED_LIB_DIR}:${STAGED_PLUGIN_DIR}${HOST_LD_LIBRARY_PATH:+:${HOST_LD_LIBRARY_PATH}}"

PLUGIN_COUNT=$(find "${STAGED_PLUGIN_DIR}" -maxdepth 1 -name '*.so' | wc -l)
STAGED_LIB_COUNT=$(find "${STAGED_LIB_DIR}"    -maxdepth 1 -type f  | wc -l)

if [[ ! -x "${STAGED_SRUN}" ]]; then
    _log "ERROR: staged srun missing at ${STAGED_SRUN}"
    exit 1
fi
if [[ "${PLUGIN_COUNT}" -eq 0 ]]; then
    _log "ERROR: no SLURM plugins staged into ${STAGED_PLUGIN_DIR}"
    exit 1
fi

_log "[V19] slurm.conf (host)   : ${HOST_SLURM_CONF}"
_log "[V19] slurm.conf (staged) : ${STAGED_SLURM_CONF}"
_log "[V19] PluginDir  (host)   : ${HOST_PLUGIN_DIR}"
_log "[V19] PluginDir  (staged) : ${STAGED_PLUGIN_DIR}"
_log "[V19] staged libs         : ${STAGED_LIB_COUNT} SLURM-specific files → ${STAGED_LIB_DIR}"
_log "[V19] staged plugins      : ${PLUGIN_COUNT} .so files"

# ── Write atomate2.yaml ───────────────────────────────────────────────────────
# Written HERE (inside the inner script) so that:
#   • VASP_STD_BIN / VASP_GAM_BIN are the real paths after module load
#   • STAGED_SRUN / LD_LIBRARY_PATH_INNER are the runtime-resolved values
#   • SRUN_FLAGS_STD / SRUN_FLAGS_GAM come from the exported environment
#   • HOST_LD_LIBRARY_PATH now includes Cray GTL (libmpi_gtl_cuda.so) path
#   • MPICH_GPU_SUPPORT_ENABLED is forwarded to VASP task ranks via --export
# The heredoc delimiter is unquoted so variables expand — which is correct
# at this point since all variables are defined in the current shell.
#
# LD_LIBRARY_PATH strategy (two separate scopes):
#
#   env LD_LIBRARY_PATH=${LD_LIBRARY_PATH_INNER}   ← for the srun PROCESS itself
#     → includes staged SLURM libs so srun can dlopen its plugins inside
#       the Shifter container (no SLURM libs in the container image).
#     → also includes HOST_LD_LIBRARY_PATH so the srun process itself
#       can find any Cray libs it needs.
#
#   --export=ALL,LD_LIBRARY_PATH=${HOST_LD_LIBRARY_PATH},...  ← for VASP TASKS
#     → srun overrides LD_LIBRARY_PATH for spawned ranks to the clean
#       module-set host path (ROCm, Cray-MPICH GTL, CUDA, MKL) with NO
#       staged SLURM libs mixed in.  Mixing staged libpmi2.so / libmunge.so
#       before the Cray PE paths causes MPI_Init to segfault on all ranks
#       because Cray-MPICH was compiled against its own specific PMI.
#     → MPICH_GPU_SUPPORT_ENABLED=1 is explicitly forwarded so Cray-MPICH
#       activates GPU transport even when the outer env might not carry it.

# Build the --export string for srun (VASP task environment)
_SRUN_EXPORT="ALL,LD_LIBRARY_PATH=${HOST_LD_LIBRARY_PATH}"
if [[ "${MPICH_GPU_SUPPORT}" == "1" ]]; then
    _SRUN_EXPORT="${_SRUN_EXPORT},MPICH_GPU_SUPPORT_ENABLED=1"
    _SRUN_EXPORT="${_SRUN_EXPORT},FI_CXI_RX_MATCH_MODE=hybrid"
fi

cat > "${ATOMATE2_CFG}" << ATOMATE2_EOF
# Atomate2-Pheasy — auto-generated on $(hostname) at $(date)
# V19 patch: Python driver runs inside Shifter; staged srun + its shared libraries
# are used to launch host NERSC VASP binaries from within the container.
VASP_CMD: env LD_LIBRARY_PATH=${LD_LIBRARY_PATH_INNER} ${STAGED_SRUN} --export=${_SRUN_EXPORT} ${SRUN_FLAGS_STD} ${VASP_STD_BIN}
VASP_GAMMA_CMD: env LD_LIBRARY_PATH=${LD_LIBRARY_PATH_INNER} ${STAGED_SRUN} --export=${_SRUN_EXPORT} ${SRUN_FLAGS_GAM} ${VASP_GAM_BIN}
ATOMATE2_EOF

_log "[V19] atomate2.yaml written : ${ATOMATE2_CFG}"
_log "[V19] VASP_CMD : env LD_LIBRARY_PATH=${LD_LIBRARY_PATH_INNER} ${STAGED_SRUN} --export=${_SRUN_EXPORT} ${SRUN_FLAGS_STD} ${VASP_STD_BIN}"
_log "[V19] MPICH_GPU_SUPPORT_ENABLED : ${MPICH_GPU_SUPPORT}"

# ── Install pinned Python package overlay on the HOST ─────────────────────────
# Per NERSC guidelines for pip+containers: install to a shared-FS target
# directory on the HOST compute node (not inside the container), then expose
# via PYTHONPATH.  The container's image filesystem is read-only and its system
# Python may not have pip; installing outside avoids both problems.
# Reference: https://docs.nersc.gov/development/languages/python/nersc-python/
#            #installing-libraries-via-pip
_log "[V19] Installing Python overlay on HOST: emmet-core==0.84.2, mp-api==0.41.2 ..."
rm -rf "${CONTAINER_PY_PKGS}/emmet"        \
       "${CONTAINER_PY_PKGS}/mp_api"       \
       "${CONTAINER_PY_PKGS}"/emmet_core-*.dist-info \
       "${CONTAINER_PY_PKGS}"/mp_api-*.dist-info 2>/dev/null || true

_PIP_OK=0
# Try pip, pip3, then python3 -m pip (in order).  One of them will be
# available from the conda env that was active when the outer script ran.
if command -v pip  &>/dev/null; then
    pip  install -q --target="${CONTAINER_PY_PKGS}" \
         --upgrade --force-reinstall --no-cache-dir --no-deps \
         "emmet-core==0.84.2" "mp-api==0.41.2" 2>&1 | tee -a "${MASTER_LOG}" \
         && _PIP_OK=1
fi
if [[ ${_PIP_OK} -eq 0 ]] && command -v pip3 &>/dev/null; then
    pip3 install -q --target="${CONTAINER_PY_PKGS}" \
         --upgrade --force-reinstall --no-cache-dir --no-deps \
         "emmet-core==0.84.2" "mp-api==0.41.2" 2>&1 | tee -a "${MASTER_LOG}" \
         && _PIP_OK=1
fi
if [[ ${_PIP_OK} -eq 0 ]]; then
    python3 -m pip install -q --target="${CONTAINER_PY_PKGS}" \
         --upgrade --force-reinstall --no-cache-dir --no-deps \
         "emmet-core==0.84.2" "mp-api==0.41.2" 2>&1 | tee -a "${MASTER_LOG}" \
         && _PIP_OK=1
fi
if [[ ${_PIP_OK} -eq 0 ]]; then
    _log "ERROR: all pip install attempts failed — cannot install emmet-core / mp-api"
    exit 1
fi

# ── Build Shifter argument list ───────────────────────────────────────────────
# PATH strategy: query the container's own default PATH (set by the image's
# ENV directive, after Shifter module injection), then PREPEND only the staged
# bin dir.  This preserves the container's conda/python paths (e.g.
# /opt/conda/bin) so the container's Python+packages are found, while ensuring
# the staged srun takes precedence over any srun that might be in the image.
#
# We do NOT use --clearenv so that SLURM_* variables (job ID, node list,
# authentication tokens, PMI endpoints) pass through to srun inside the
# container.  We explicitly clear CONDA_PREFIX / CONDA_DEFAULT_ENV to prevent
# the HOST's active conda env from interfering with the container's Python.

_log "[V19] Querying container's native PATH ..."
_CPATH=$(
    "${SHIFTER_BIN}" "${SHIFTER_MODULE}" --image="${CONTAINER_IMAGE}" \
        /bin/sh -c 'echo $PATH' 2>/dev/null || true
)
_CPATH="${_CPATH:-/opt/conda/bin:/usr/local/bin:/usr/bin:/bin}"
EFFECTIVE_PATH="${STAGED_BIN_DIR}:${_CPATH}"
_log "[V19] Container native PATH : ${_CPATH}"
_log "[V19] Effective PATH        : ${EFFECTIVE_PATH}"

SHIFTER_ARGS=(
    "${SHIFTER_MODULE}"
    --image="${CONTAINER_IMAGE}"
    --env=PATH="${EFFECTIVE_PATH}"
    --env=LD_LIBRARY_PATH="${LD_LIBRARY_PATH_INNER}"
    --env=SLURM_CONF="${STAGED_SLURM_CONF}"
    --env=JOBFLOW_CONFIG_FILE="${JOBFLOW_CFG}"
    --env=ATOMATE2_CONFIG_FILE="${ATOMATE2_CFG}"
    --env=PMG_VASP_PSP_DIR="${POTCAR_DIR}"
    --env=NERSC_RUNTIME="${NERSC_RUNTIME_DIR}"
    --env=PYTHONPATH="${CONTAINER_PY_PKGS}"
    --env=PYTHONNOUSERSITE=1
    --env=OMP_NUM_THREADS="${OMP_NUM_THREADS}"
    --env=OMP_PLACES="${OMP_PLACES}"
    --env=OMP_PROC_BIND=spread
    --env=FI_CXI_RX_MATCH_MODE=hybrid
    --env=SLURM_NNODES="${SLURM_NNODES}"   # ← NEW: lets Python set NPAR=SLURM_NNODES
)

# ── FIX: Add MPICH_GPU_SUPPORT_ENABLED for GPU runs ──────────────────────────
# ROOT CAUSE OF SEGFAULT #3:
# Without MPICH_GPU_SUPPORT_ENABLED=1, Cray-MPICH does not activate the GPU
# transport layer (libmpi_gtl_cuda.so) for device-buffer MPI operations.
# VASP's ScaLAPACK and FFT collectives use device buffers on GPU runs.
if [[ "${MPICH_GPU_SUPPORT}" == "1" ]]; then
    SHIFTER_ARGS+=(--env=MPICH_GPU_SUPPORT_ENABLED=1)
    _log "[V19] MPICH_GPU_SUPPORT_ENABLED=1 added to Shifter env"
fi

# ── Smoke-test: verify staged srun loads cleanly inside the container ─────────
# Use the absolute path to the staged srun binary — no PATH dependency.
_log "[V19] Smoke-testing staged srun inside Shifter..."
if ! "${SHIFTER_BIN}" "${SHIFTER_ARGS[@]}" \
        "${STAGED_SRUN}" --version 2>&1 | tee -a "${MASTER_LOG}"; then
    _log "ERROR: staged srun smoke-test failed — check LD_LIBRARY_PATH and PluginDir staging"
    exit 1
fi

# ── Verify imports inside the container ───────────────────────────────────────
_log "[V19] Verifying Shifter-side Python imports..."
"${SHIFTER_BIN}" "${SHIFTER_ARGS[@]}" python3 -s - <<'PYCHECK'
from importlib.metadata import version
import os, shutil, sys

print("python        :", sys.executable)
print("python prefix :", sys.prefix)
print("CONDA_PREFIX  :", os.environ.get("CONDA_PREFIX", "(not set)"))
print("emmet-core    :", version("emmet-core"))
print("mp-api        :", version("mp-api"))
print("pymatgen      :", version("pymatgen"))
print("SLURM_CONF    :", os.environ.get("SLURM_CONF"))
print("which srun    :", shutil.which("srun"))
print("SLURM_NNODES  :", os.environ.get("SLURM_NNODES", "(not set)"))
print("MPICH_GPU     :", os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "(not set)"))

conda_leaks = [p for p in sys.path if "/.conda/envs/" in p]
print("conda leaks   :", conda_leaks[:5] if conda_leaks else "none")
if conda_leaks:
    raise SystemExit("Host conda paths leaked into the Shifter Python environment")
print("[V19] Import check PASSED")
PYCHECK

# ── Launch Python workflow driver inside Shifter ──────────────────────────────
_log "[V19] Launching workflow driver inside Shifter (isolated env + staged srun)..."

STDERR_LOG="${LOG_DIR}/stderr_${MPID}_${TIMESTAMP}.err"

# Temporarily disable 'set -e' around the workflow launch so we can
# capture the exit code and still print the timing summary.
#set +e
"${SHIFTER_BIN}" "${SHIFTER_ARGS[@]}" \
    python3 -s "${PY_BOOTSTRAP}" \
        --mpid        "${MPID}"       \
        --mp-api-key  "${MP_API_KEY}" \
        --docs-store  "${DOCS_STORE}" \
        --blob-store  "${BLOB_STORE}" \
        --logdir      "${LOG_DIR}"    \
    2> >(tee -a "${STDERR_LOG}" >&2) \
    | tee -a "${MASTER_LOG}"

WF_EXIT=${PIPESTATUS[0]}
#set -e

# ── Timing summary ────────────────────────────────────────────────────────────
WF_END=$(date +%s)
TOTAL_SEC=$((WF_END - WF_START))
TOTAL_H=$((TOTAL_SEC / 3600))
TOTAL_M=$(( (TOTAL_SEC % 3600) / 60 ))
TOTAL_S=$((TOTAL_SEC % 60))

_log ""
_log "════════════════════════════════════════════════════════════════"
if [[ ${WF_EXIT} -eq 0 ]]; then
    _log "  ✔  WORKFLOW COMPLETE"
else
    _log "  ✗  WORKFLOW FAILED  (exit code: ${WF_EXIT})"
fi
_log "  MP-ID    : ${MPID}"
_log "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
_log "  Wall     : ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
_log "  Log      : ${MASTER_LOG}"
_log "════════════════════════════════════════════════════════════════"

exit ${WF_EXIT}
INNER_EOF

chmod +x "$INNER_SCRIPT"

# ─── Submit via salloc ────────────────────────────────────────────────────────

echo ""
echo "[$(date '+%H:%M:%S')] Requesting interactive allocation..."
echo "  → salloc -N ${N_NODES} -C ${CONSTRAINT} -A ${SLURM_ACCOUNT} -t 4:00:00 --qos=interactive"
echo ""

salloc \
    -N "$N_NODES"       \
    -C "$CONSTRAINT"    \
    -A "$SLURM_ACCOUNT" \
    -t 4:00:00          \
    --qos=interactive   \
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
