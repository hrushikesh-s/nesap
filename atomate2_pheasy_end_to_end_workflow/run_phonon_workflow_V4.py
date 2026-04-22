#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomate2-Pheasy End-to-End Phonon Workflow Driver  (V4)
-------------------------------------------------------
Execution context
-----------------
This script runs INSIDE the Shifter container on a NERSC Perlmutter compute
node.  It is invoked by submit_atomate2_phonon_wf_shifter_V19.sh via a thin
bootstrap shim that imports and calls main().

Architecture
------------
- Python workflow driver  →  runs inside Shifter
- VASP (vasp_std/vasp_gam)  →  runs on the HOST via a staged host srun binary
- The staged srun is on PATH inside the container (${NERSC_RUNTIME}/bin)
- VASP_CMD / VASP_GAMMA_CMD are written to ATOMATE2_CONFIG_FILE by the outer
  shell script after the VASP module is loaded on the compute node

Pre-flight checks (V4)
-----------------------
- Verifies the staged srun is on PATH (inside the container)
- Validates ATOMATE2_CONFIG_FILE, JOBFLOW_CONFIG_FILE, PMG_VASP_PSP_DIR
- Prints all config values so problems are visible before VASP is invoked

Per-job timing
--------------
Captured by hooking into jobflow's logging stream.
jobflow emits:
    "Starting job - {name} ({uuid})"
    "Finished job - {name} ({uuid})"
All output is tee'd to both the terminal and .out / .err files.

NPAR fix (V4 patch)
-------------------
NPAR is now computed dynamically from SLURM_NNODES (== number of GPU nodes ==
number of GPUs / 4 on Perlmutter).  Setting NPAR = N_NODES satisfies the VASP
constraint  KPAR × NPAR × NCORE == N_MPI_ranks  with NCORE=1, KPAR=N_TASKS/NPAR=4.
Hard-coding NPAR=12 caused all ranks to segfault on small structures (e.g. Si
mp-149) because 16 MPI ranks cannot be divided evenly into 12 band groups.
"""

import argparse
import logging
import os
import sys
import time
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
#  TIMED LOGGING HANDLER
#  Hooks into jobflow's logger to capture per-job start/end timestamps.
# ═════════════════════════════════════════════════════════════════════════════

class JobTimingHandler(logging.Handler):
    def __init__(self, out_fh, err_fh):
        super().__init__()
        self.out_fh          = out_fh
        self.err_fh          = err_fh
        self.job_timings     = []   # [(step_name, uuid, elapsed_seconds)]
        self._current_name   = None
        self._current_uuid   = None
        self._current_step   = 0
        self._step_start     = None
        self._workflow_start = time.perf_counter()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _tee(self, text, is_err=False):
        """Write to terminal AND log file simultaneously."""
        print(text, flush=True)
        self.out_fh.write(text + "\n")
        self.out_fh.flush()
        if is_err:
            self.err_fh.write(text + "\n")
            self.err_fh.flush()

    @staticmethod
    def _fmt_elapsed(seconds):
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"

    # ── emit ─────────────────────────────────────────────────────────────────

    def emit(self, record):
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()

        # ── Job start ────────────────────────────────────────────────────
        if msg.startswith("Starting job"):
            self._current_step += 1
            self._step_start    = time.perf_counter()
            try:
                rest                = msg.split(" - ", 1)[1]        # "{name} ({uuid})"
                self._current_name  = rest.split(" (")[0]
                self._current_uuid  = rest.split("(")[1].rstrip(")")
            except (IndexError, ValueError):
                self._current_name  = msg
                self._current_uuid  = "unknown"

            sep    = "=" * 68
            banner = (
                f"\n{sep}\n"
                f"  ▶  STEP {self._current_step} START  |  {ts}\n"
                f"  Job  : {self._current_name}\n"
                f"  UUID : {self._current_uuid}\n"
                f"{sep}"
            )
            self._tee(banner)

        # ── Job end ──────────────────────────────────────────────────────
        elif msg.startswith("Finished job") and self._step_start is not None:
            elapsed = time.perf_counter() - self._step_start
            self.job_timings.append(
                (self._current_name, self._current_uuid, elapsed)
            )
            sep    = "=" * 68
            banner = (
                f"{sep}\n"
                f"  ✔  STEP {self._current_step} END    |  {ts}\n"
                f"  Job     : {self._current_name}\n"
                f"  Elapsed : {self._fmt_elapsed(elapsed)}\n"
                f"{sep}\n"
            )
            self._tee(banner)
            self._step_start = None

        # ── Warnings / errors → also go to .err ──────────────────────────
        elif record.levelno >= logging.WARNING:
            line = f"[{ts}] [{record.levelname}] {msg}"
            self._tee(line, is_err=True)

        # ── Regular info ─────────────────────────────────────────────────
        else:
            self._tee(f"[{ts}] {msg}")

    # ── summary table ─────────────────────────────────────────────────────────

    def print_summary(self, mpid: str):
        total = time.perf_counter() - self._workflow_start

        W = 68

        def row(name, status, wall):
            return f"║ {name:<32} ║ {status:^10} ║ {wall:>16} ║"

        lines = [
            "",
            "╔" + "═" * (W - 2) + "╗",
            f"║{'  WORKFLOW TIMING SUMMARY  —  ' + mpid:^{W-2}}║",
            "╠" + "═" * 34 + "╦" + "═" * 12 + "╦" + "═" * 18 + "╣",
            row("Step", "Status", "Wall Time"),
            "╠" + "═" * 34 + "╬" + "═" * 12 + "╬" + "═" * 18 + "╣",
        ]
        for name, _, elapsed in self.job_timings:
            lines.append(row(name[:32], "✔ DONE", self._fmt_elapsed(elapsed)))

        lines += [
            "╠" + "═" * 34 + "╬" + "═" * 12 + "╬" + "═" * 18 + "╣",
            row("TOTAL", "✔ DONE", self._fmt_elapsed(total)),
            "╚" + "═" * 34 + "╩" + "═" * 12 + "╩" + "═" * 18 + "╝",
            "",
        ]
        summary = "\n".join(lines)
        print(summary, flush=True)
        self.out_fh.write(summary + "\n")
        self.out_fh.flush()


# ═════════════════════════════════════════════════════════════════════════════
#  PRE-FLIGHT CHECK  (V4)
#  Validates the Shifter-side launch configuration BEFORE any MP/VASP work
#  so that configuration problems are immediately visible rather than buried
#  in a custodian traceback.
#
#  Expected environment (set by V19.sh inner script via Shifter --env flags):
#    PATH               : includes ${NERSC_RUNTIME}/bin (staged srun)
#    ATOMATE2_CONFIG_FILE : atomate2.yaml with VASP_CMD / VASP_GAMMA_CMD
#    JOBFLOW_CONFIG_FILE  : jobflow.yaml with JSONStore paths
#    PMG_VASP_PSP_DIR     : POTCAR directory
#    SLURM_CONF           : staged slurm.conf (PluginDir → staged plugins)
# ═════════════════════════════════════════════════════════════════════════════

def preflight_slurm_check():
    """
    Validate the Shifter-side launch configuration.

    Checks:
      1. staged srun is on PATH (inside the container)
      2. ATOMATE2_CONFIG_FILE exists and contains VASP_CMD + VASP_GAMMA_CMD
      3. JOBFLOW_CONFIG_FILE exists
      4. PMG_VASP_PSP_DIR is set (warning only — POTCAR dir may be on scratch)
      5. SLURM_CONF points to the staged slurm.conf
      6. Print SLURM allocation info and NPAR being used

    Raises RuntimeError on any fatal misconfiguration.
    """
    import shutil

    SEP = "─" * 68
    print(f"\n{SEP}")
    print("  SHIFTER PRE-FLIGHT: launch configuration check  (V4)")
    print(SEP)

    fatal = False

    # ── 1. Staged srun on PATH ────────────────────────────────────────────
    path_srun = shutil.which("srun")
    if path_srun:
        print(f"  ✔  srun             : {path_srun}")
    else:
        print("  ✗  srun             : not found in PATH")
        print("     Expected the staged srun in ${NERSC_RUNTIME}/bin to be on PATH.")
        fatal = True

    # ── 2. ATOMATE2_CONFIG_FILE ───────────────────────────────────────────
    a2_cfg            = os.environ.get("ATOMATE2_CONFIG_FILE", "")
    vasp_cmd_found    = False
    vasp_gamma_found  = False

    if a2_cfg and Path(a2_cfg).is_file():
        print(f"  ✔  ATOMATE2_CONFIG  : {a2_cfg}")
        try:
            print("     Contents:")
            with open(a2_cfg) as fh:
                for line in fh:
                    line = line.rstrip()
                    if line and not line.startswith("#"):
                        print(f"       {line}")
                    if line.startswith("VASP_CMD:"):
                        vasp_cmd_found = True
                    if line.startswith("VASP_GAMMA_CMD:"):
                        vasp_gamma_found = True
        except OSError as exc:
            print(f"  ✗  ATOMATE2_CONFIG  : read error: {exc}")
            fatal = True
    else:
        print(f"  ✗  ATOMATE2_CONFIG  : {a2_cfg or '(not set)'}")
        fatal = True

    if not vasp_cmd_found:
        print("  ✗  VASP_CMD         : missing from ATOMATE2_CONFIG_FILE")
        fatal = True
    else:
        print("  ✔  VASP_CMD         : present")

    if not vasp_gamma_found:
        print("  ✗  VASP_GAMMA_CMD   : missing from ATOMATE2_CONFIG_FILE")
        fatal = True
    else:
        print("  ✔  VASP_GAMMA_CMD   : present")

    # ── 3. JOBFLOW_CONFIG_FILE ────────────────────────────────────────────
    jf_cfg = os.environ.get("JOBFLOW_CONFIG_FILE", "")
    if jf_cfg and Path(jf_cfg).is_file():
        print(f"  ✔  JOBFLOW_CONFIG   : {jf_cfg}")
    else:
        print(f"  ✗  JOBFLOW_CONFIG   : {jf_cfg or '(not set)'}")
        fatal = True

    # ── 4. POTCAR root ────────────────────────────────────────────────────
    psp_dir = os.environ.get("PMG_VASP_PSP_DIR", "")
    if psp_dir and Path(psp_dir).exists():
        print(f"  ✔  PMG_VASP_PSP_DIR : {psp_dir}")
    else:
        print(f"  ⚠  PMG_VASP_PSP_DIR : {psp_dir or '(not set)'}")

    # ── 5. SLURM_CONF ─────────────────────────────────────────────────────
    slurm_conf = os.environ.get("SLURM_CONF", "")
    if slurm_conf and Path(slurm_conf).is_file():
        print(f"  ✔  SLURM_CONF       : {slurm_conf}")
    elif slurm_conf:
        print(f"  ⚠  SLURM_CONF       : {slurm_conf}  (file not found)")
    else:
        print("  ⚠  SLURM_CONF       : (not set)")

    # ── 6. NPAR / allocation info ─────────────────────────────────────────
    n_nodes = int(os.environ.get("SLURM_NNODES", "0"))
    npar    = _compute_npar()
    print(f"  ✔  SLURM_NNODES     : {n_nodes if n_nodes else '(not set — defaulting to 1)'}")
    print(f"  ✔  NPAR (dynamic)   : {npar}  (= SLURM_NNODES, satisfies KPAR×NPAR×NCORE=N_MPI)")
    print(f"  ✔  MPICH_GPU_SUPPORT: {os.environ.get('MPICH_GPU_SUPPORT_ENABLED', '(not set)')}")

    print(SEP)

    if fatal:
        raise RuntimeError(
            "Shifter pre-flight FAILED — see ✗ lines above.\n"
            "Common causes:\n"
            "  • staged srun is missing from ${NERSC_RUNTIME}/bin — "
              "check srun staging in the inner script\n"
            "  • ATOMATE2_CONFIG_FILE not set or empty — "
              "check that the outer script wrote atomate2.yaml\n"
            "  • JOBFLOW_CONFIG_FILE not set or empty\n"
        )

    print("  ✔  All pre-flight checks passed — proceeding with workflow\n")


# ═════════════════════════════════════════════════════════════════════════════
#  NPAR HELPER
#  ─────────────────────────────────────────────────────────────────────────
#  On Perlmutter GPU (4 GPUs/node, NCORE=1, 1 MPI rank per GPU):
#
#    N_MPI = SLURM_NNODES × 4
#    VASP requires:  KPAR × NPAR × NCORE = N_MPI
#    With NCORE=1, KPAR=4:  NPAR = SLURM_NNODES  → always integer ✔
#
#  OLD behaviour (NPAR=12 hard-coded):
#    4 nodes → 16 MPI ranks:  16 / 12 = 1.33 → NOT integer → segfault
#    8 nodes → 32 MPI ranks:  32 / 12 = 2.67 → NOT integer → often silent
#    crash, lucky auto-adjust for large NBANDS
#
#  NEW behaviour: NPAR = SLURM_NNODES (read at runtime from env)
#    4 nodes → NPAR=4  → 16/4=4 ranks/group → KPAR=4 ✔
#    8 nodes → NPAR=8  → 32/8=4 ranks/group → KPAR=4 ✔
# ═════════════════════════════════════════════════════════════════════════════

def _compute_npar() -> int:
    """
    Return NPAR for the current SLURM allocation.

    NPAR = SLURM_NNODES  (one band group per node).
    Falls back to 1 if SLURM_NNODES is not set (local testing).
    """
    n_nodes = int(os.environ.get("SLURM_NNODES", "1"))
    return max(1, n_nodes)


# ═════════════════════════════════════════════════════════════════════════════
#  WORKFLOW BUILDER
#  ⚡ Kept in exact parity with Hrushikesh's production submission script.
#     Do NOT change INCAR tags, PhononMaker kwargs, or metadata keys
#     without also updating the production script.
# ═════════════════════════════════════════════════════════════════════════════

RUN_TAG  = "fresh phonon submission from MP triclinic filter"
CATEGORY = "fresh_mp_triclinic_nm"


def build_phonon_flow(mpid: str, structure):
    """
    Build the Atomate2-Pheasy PhononMaker flow.

    Exact parity with production script:
      - PhononMaker kwargs
      - INCAR overrides via update_user_incar_settings
      - Metadata attachment via add_metadata_to_flow
      - Per-job metadata updates via flow.update_metadata

    NPAR is computed dynamically from SLURM_NNODES so that
    KPAR × NPAR × NCORE always equals the actual MPI rank count.
    Hard-coding NPAR=12 caused all-rank segfaults for small structures
    like Si (mp-149) running on 4 nodes (16 MPI ranks).
    """
    from atomate2.vasp.flows.pheasy import PhononMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.common.flows.pheasy import BasePhononMaker
    from atomate2.common.powerups import add_metadata_to_flow
    from atomate2.vasp.powerups import update_user_incar_settings

    # ── Compute NPAR from the live SLURM allocation ──────────────────────
    npar = _compute_npar()

    # ── 1. Build PhononMaker ─────────────────────────────────────────────
    maker = PhononMaker(
        mp_id=mpid,
        cal_anhar_fcs=False,
        use_symmetrized_structure="primitive",
        min_length=12.0,
        force_90_degrees=True,
        force_diagonal=True,
    )

    flow = maker.make(structure=structure)

    # ── 2. VASP INCAR overrides ──────────────────────────────────────────
    #
    #  NPAR NOTE:
    #  Previously hard-coded as 12.  This crashed all 16 MPI ranks on 4 nodes
    #  because VASP requires KPAR × NPAR × NCORE = N_MPI_ranks (NCORE=1).
    #  16 / 12 is not an integer, so VASP auto-adjusts to 16 groups but then
    #  segfaults during band distribution for small-NBANDS structures.
    #  Solution: NPAR = SLURM_NNODES (4 for 4 nodes) → 16/4 = 4 ranks/group.
    #
    flow = update_user_incar_settings(
        flow,
        {
            "ALGO":     "Normal",
            "ENCUT":    600,
            "ISMEAR":   0,
            "SIGMA":    0.05,
            "KSPACING": 0.15,
            "ISPIN":    1,
            "EDIFFG":   -1e-04,
            "EDIFF":    1e-07,
            "NPAR":     npar,   # ← dynamic: SLURM_NNODES (was hard-coded 12)
        },
    )

    # ── 3. Flow-level metadata ───────────────────────────────────────────
    flow = add_metadata_to_flow(
        flow,
        {
            "mp_id":           mpid,
            "run_tag":         RUN_TAG,
            "category":        CATEGORY,
            "submission_mode": "fresh",
            "kspacing":        "accurate(0.15)",
            "min_length":      12.0,
        },
        class_filter=(BaseVaspMaker, BasePhononMaker, PhononMaker),
    )

    # ── 4. Per-job metadata ──────────────────────────────────────────────
    for name_filter in (
        "get_supercell_size",
        "generate_frequencies_eigenvectors",
        "Force field",
        "generate_phonon_displacements",
        "run_phonon_displacements",
        "structure_to_primitive",
    ):
        with suppress(Exception):
            flow.update_metadata(
                {"mp_id": mpid, "category": CATEGORY},
                name_filter=name_filter,
                dynamic=False,
            )

    return flow


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Atomate2-Pheasy end-to-end phonon workflow (run_locally, Shifter mode)"
    )
    parser.add_argument("--mpid",       required=True, help="e.g. mp-149")
    parser.add_argument("--mp-api-key", required=True, help="Materials Project API key")
    parser.add_argument("--docs-store", required=True, help="Path for docs JSONStore (.json)")
    parser.add_argument("--blob-store", required=True, help="Path for blob JSONStore (.json)")
    parser.add_argument("--logdir",     default=".",   help="Directory for .out / .err logs")
    args = parser.parse_args()

    # ── Set up log files ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir    = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    out_path  = logdir / f"workflow_{args.mpid}_{timestamp}.out"
    err_path  = logdir / f"workflow_{args.mpid}_{timestamp}.err"

    print(f"\n{'='*68}")
    print(f"  Atomate2-Pheasy Phonon Workflow  (run_locally, Shifter mode, V4)")
    print(f"  MP-ID      : {args.mpid}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Docs store : {args.docs_store}")
    print(f"  Blob store : {args.blob_store}")
    print(f"  Log (.out) : {out_path}")
    print(f"  Log (.err) : {err_path}")
    print(f"  SLURM_NNODES: {os.environ.get('SLURM_NNODES', '(not set)')}")
    print(f"  NPAR (dyn) : {_compute_npar()}")
    print(f"{'='*68}\n")

    out_fh = open(out_path, "w")
    err_fh = open(err_path, "w")

    # ── Attach timed handler to jobflow's logger ───────────────────────────
    timing_handler = JobTimingHandler(out_fh, err_fh)
    timing_handler.setFormatter(logging.Formatter("%(message)s"))
    jf_logger = logging.getLogger("jobflow")
    jf_logger.setLevel(logging.INFO)
    jf_logger.addHandler(timing_handler)

    try:
        # ── Step 0: Pre-flight check ───────────────────────────────────────
        # Validates the Shifter-side environment (staged srun, config files)
        # before any MP/VASP work so problems are immediately visible.
        preflight_slurm_check()

        # ── Step 1: Fetch structure from Materials Project ─────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Fetching structure for {args.mpid} from Materials Project...")
        from mp_api.client import MPRester
        with MPRester(args.mp_api_key, use_document_model=False) as mpr:
            structure = mpr.get_structure_by_material_id(args.mpid)
        if structure is None:
            raise RuntimeError(
                f"Could not fetch structure for {args.mpid} from Materials Project."
            )
        sg_symbol, sg_number = structure.get_space_group_info()
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{ts}] Structure: {structure.formula}  |  "
            f"{len(structure)} sites  |  "
            f"spacegroup: {sg_symbol} ({sg_number})"
        )

        # ── Step 2: Build flow ─────────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Building PhononMaker flow...")
        flow = build_phonon_flow(args.mpid, structure)
        n_jobs = len(list(flow.jobs))
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Flow built — {n_jobs} jobs queued")

        # ── Step 3: Configure jobflow stores (JSONStore, no MongoDB) ────────
        from jobflow import JobStore
        from maggma.stores import JSONStore

        Path(args.docs_store).parent.mkdir(parents=True, exist_ok=True)
        Path(args.blob_store).parent.mkdir(parents=True, exist_ok=True)

        # Reinitialise empty/corrupt JSON files before connecting
        for store_path in (Path(args.docs_store), Path(args.blob_store)):
            if store_path.exists() and store_path.stat().st_size == 0:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] WARNING: {store_path} is empty; reinitialising as []")
                store_path.write_text("[]", encoding="utf-8")

        docs_store = JSONStore(args.docs_store, read_only=False)
        blob_store = JSONStore(args.blob_store, read_only=False)
        store      = JobStore(docs_store, additional_stores={"data": blob_store})

        # ── Step 4: Run workflow ───────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Launching workflow via run_locally...\n")

        from jobflow import run_locally
        run_locally(
            flow,
            store=store,
            log=True,
            create_folders=True,
            ensure_success=True,
        )

        # ── Step 5: Timing summary ─────────────────────────────────────────
        timing_handler.print_summary(args.mpid)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] ✔ Workflow completed successfully!")
        print(f"  Results  → {args.docs_store}")
        print(f"  Blob     → {args.blob_store}")
        print(f"  Full log → {out_path}")
        print(f"  Err  log → {err_path}")

    except Exception:
        ts  = datetime.now().strftime("%H:%M:%S")
        tb  = traceback.format_exc()
        msg = f"[{ts}] ✗ WORKFLOW FAILED:\n{tb}"
        print(msg, flush=True)
        err_fh.write(msg + "\n")
        err_fh.flush()
        sys.exit(1)

    finally:
        out_fh.close()
        err_fh.close()


if __name__ == "__main__":
    main()
