#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomate2-Pheasy End-to-End Phonon Workflow Driver
--------------------------------------------------
Runs INSIDE the Shifter container via jobflow's run_locally.
Called by submit_atomate2_phonon_wf_shifter.sh.

PhononMaker config, VASP INCAR tags, and metadata are kept in exact
parity with Hrushikesh's production submission script.

Per-job timing is captured by hooking into jobflow's logging stream.
All output is tee'd to both terminal (visible in tmux) and .out/.err files.

V8 changes
----------
- Added pre-flight check that logs SLURM_CONF value, verifies the staged
  srun binary exists, reads PluginDir from the staged conf, and confirms
  the PluginDir path is reachable — all BEFORE run_locally starts.
  This makes conf problems immediately visible rather than buried inside
  a custodian traceback.
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
#  jobflow emits:
#    "Starting job - {name} ({uuid})"
#    "Finished job - {name} ({uuid})"
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

    # ── summary table ─────────────────────────────────────────────────────

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
#  V8 PRE-FLIGHT CHECK
#  Verifies the SLURM/srun environment BEFORE run_locally starts so that
#  conf problems are immediately visible rather than buried in a custodian
#  traceback.  Prints a ✔/✗ line for each check; raises on fatal issues.
# ═════════════════════════════════════════════════════════════════════════════

def preflight_slurm_check():
    """
    Inspect SLURM_CONF, the staged srun binary, and the PluginDir it points
    to.  Logs a clear ✔/✗ summary so any remaining config problem is
    immediately obvious without digging through custodian tracebacks.

    Raises RuntimeError on fatal issues (missing srun, wrong PluginDir).
    """
    SEP = "─" * 68
    print(f"\n{SEP}")
    print("  V8 PRE-FLIGHT: SLURM / srun environment check")
    print(SEP)

    fatal = False

    # ── 1. SLURM_CONF ────────────────────────────────────────────────────
    slurm_conf = os.environ.get("SLURM_CONF", "")
    if slurm_conf:
        conf_path = Path(slurm_conf)
        if conf_path.is_file():
            print(f"  ✔  SLURM_CONF       : {slurm_conf}  (file exists)")
        else:
            print(f"  ✗  SLURM_CONF       : {slurm_conf}  ← FILE NOT FOUND")
            fatal = True
    else:
        print("  ⚠  SLURM_CONF       : (not set — srun will use compiled-in default)")
        slurm_conf = None

    # ── 2. PluginDir value in staged conf ────────────────────────────────
    plugin_dir_val = "(could not read)"
    if slurm_conf and Path(slurm_conf).is_file():
        try:
            with open(slurm_conf) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("PluginDir="):
                        plugin_dir_val = line.split("=", 1)[1]
                        break
                else:
                    plugin_dir_val = "(no PluginDir= line — srun uses compiled-in default)"
        except OSError as exc:
            plugin_dir_val = f"(read error: {exc})"

        if plugin_dir_val.startswith("("):
            # No explicit line — srun will use compiled-in /usr/lib64/slurm
            print(f"  ✗  PluginDir        : {plugin_dir_val}")
            print("       → srun will fall back to /usr/lib64/slurm inside the container")
            print("       → This is the V8 bug target. Check that the bash patch ran.")
            fatal = True
        else:
            pdir = Path(plugin_dir_val)
            if pdir.is_dir():
                so_count = len(list(pdir.glob("*.so")))
                print(f"  ✔  PluginDir        : {plugin_dir_val}  ({so_count} .so files)")
                if so_count == 0:
                    print("       ⚠  WARNING: PluginDir exists but contains no .so files!")
                    print("          srun may still fail. Check the staging step in the shell script.")
            else:
                print(f"  ✗  PluginDir        : {plugin_dir_val}  ← DIRECTORY NOT FOUND")
                fatal = True

    # ── 3. Staged srun binary ────────────────────────────────────────────
    nersc_runtime = os.environ.get("NERSC_RUNTIME",
                                    "/pscratch/sd/h/hrushi99/nersc_runtime")
    staged_srun = Path(nersc_runtime) / "bin" / "srun"
    if staged_srun.is_file():
        print(f"  ✔  Staged srun      : {staged_srun}")
    else:
        # Fall back to PATH
        import shutil
        path_srun = shutil.which("srun")
        if path_srun:
            print(f"  ⚠  Staged srun      : {staged_srun} not found")
            print(f"       → Falling back to PATH srun: {path_srun}")
        else:
            print(f"  ✗  srun             : not found at {staged_srun} or in PATH")
            fatal = True

    # ── 4. ATOMATE2_CONFIG_FILE ──────────────────────────────────────────
    a2_cfg = os.environ.get("ATOMATE2_CONFIG_FILE", "")
    if a2_cfg and Path(a2_cfg).is_file():
        print(f"  ✔  ATOMATE2_CONFIG  : {a2_cfg}")
        try:
            print("     Contents:")
            with open(a2_cfg) as fh:
                for line in fh:
                    line = line.rstrip()
                    if line and not line.startswith("#"):
                        print(f"       {line}")
        except OSError:
            pass
    else:
        val = a2_cfg or "(not set)"
        print(f"  ⚠  ATOMATE2_CONFIG  : {val}")

    # ── 5. JOBFLOW_CONFIG_FILE ───────────────────────────────────────────
    jf_cfg = os.environ.get("JOBFLOW_CONFIG_FILE", "")
    if jf_cfg and Path(jf_cfg).is_file():
        print(f"  ✔  JOBFLOW_CONFIG   : {jf_cfg}")
    else:
        val = jf_cfg or "(not set)"
        print(f"  ⚠  JOBFLOW_CONFIG   : {val}")

    print(SEP)

    if fatal:
        raise RuntimeError(
            "V8 pre-flight failed: SLURM/srun environment is not correctly configured.\n"
            "Check the output above for ✗ lines.  The most common cause:\n"
            "  • slurm.conf has no PluginDir= line and the V8 append step did not run\n"
            "  • PluginDir directory doesn't exist or has no .so files\n"
            "Re-run the shell script and look for '[V8] PluginDir line was ABSENT' in the log."
        )

    print("  ✔  All pre-flight checks passed — proceeding with workflow\n")


# ═════════════════════════════════════════════════════════════════════════════
#  WORKFLOW BUILDER
#  ⚡ Kept in exact parity with Hrushikesh's production submission script.
#     Do NOT change INCAR tags, PhononMaker kwargs, or metadata keys
#     without also updating the production script.
# ═════════════════════════════════════════════════════════════════════════════

# Workflow metadata constants (mirrors production script)
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
    """
    from atomate2.vasp.flows.pheasy import PhononMaker
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.common.flows.pheasy import BasePhononMaker
    from atomate2.common.powerups import add_metadata_to_flow
    from atomate2.vasp.powerups import update_user_incar_settings

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
    flow = update_user_incar_settings(
        flow,
        {
            "ALGO":   "Normal",
            "ENCUT":  600,
            "ISMEAR": 0,
            "SIGMA":  0.05,
            "KSPACING": 0.15,
            "ISPIN":  1,
            "EDIFFG": -1e-04,
            "EDIFF":  1e-07,
            "NPAR":   12,
        },
    )

    # ── 3. Flow-level metadata ───────────────────────────────────────────
    flow = add_metadata_to_flow(
        flow,
        {
            "mp_id":            mpid,
            "run_tag":          RUN_TAG,
            "category":         CATEGORY,
            "submission_mode":  "fresh",
            "kspacing":         "accurate(0.15)",
            "min_length":       12.0,
        },
        class_filter=(BaseVaspMaker, BasePhononMaker, PhononMaker),
    )

    # ── 4. Per-job metadata (mirrors production suppress-wrapped calls) ───
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
        description="Atomate2-Pheasy end-to-end phonon workflow (run_locally)"
    )
    parser.add_argument("--mpid",       required=True, help="e.g. mp-996970")
    parser.add_argument("--mp-api-key", required=True, help="Materials Project API key")
    parser.add_argument("--docs-store", required=True, help="Path for docs JSONStore output (.json)")
    parser.add_argument("--blob-store", required=True, help="Path for blob JSONStore output (.json)")
    parser.add_argument("--logdir",     default=".",   help="Directory for .out / .err logs")
    args = parser.parse_args()

    # ── Set up log files ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir    = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    out_path  = logdir / f"workflow_{args.mpid}_{timestamp}.out"
    err_path  = logdir / f"workflow_{args.mpid}_{timestamp}.err"

    print(f"\n{'='*68}")
    print(f"  Atomate2-Pheasy Phonon Workflow  (run_locally)  [V8]")
    print(f"  MP-ID      : {args.mpid}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Docs store : {args.docs_store}")
    print(f"  Blob store : {args.blob_store}")
    print(f"  Log (.out) : {out_path}")
    print(f"  Log (.err) : {err_path}")
    print(f"{'='*68}\n")

    out_fh = open(out_path, "w")
    err_fh = open(err_path, "w")

    # ── Attach timed handler to jobflow's logger ──────────────────────────
    timing_handler = JobTimingHandler(out_fh, err_fh)
    timing_handler.setFormatter(logging.Formatter("%(message)s"))
    jf_logger = logging.getLogger("jobflow")
    jf_logger.setLevel(logging.INFO)
    jf_logger.addHandler(timing_handler)

    try:
        # ── V8 Step 0: Pre-flight SLURM/srun check ────────────────────────
        # Runs before any MP/VASP work so problems are immediately visible.
        preflight_slurm_check()

        # ── Step 1: Fetch structure from Materials Project ────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Fetching structure for {args.mpid} from Materials Project...")
        from mp_api.client import MPRester
        #with MPRester(args.mp_api_key) as mpr:
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

        # ── Step 2: Build flow ────────────────────────────────────────────
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Building PhononMaker flow...")
        flow = build_phonon_flow(args.mpid, structure)
        n_jobs = len(list(flow.jobs))
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Flow built — {n_jobs} jobs queued")

        # ── Step 3: Configure jobflow stores (JSONStore, no MongoDB) ──────
        from jobflow import JobStore
        from maggma.stores import JSONStore

#        # Ensure parent directories exist
#        Path(args.docs_store).parent.mkdir(parents=True, exist_ok=True)
#        Path(args.blob_store).parent.mkdir(parents=True, exist_ok=True)
#
#        docs_store = JSONStore(args.docs_store, read_only=False)
#        blob_store = JSONStore(args.blob_store, read_only=False)
#        store      = JobStore(docs_store, additional_stores={"data": blob_store})



        # Ensure parent directories exist
        Path(args.docs_store).parent.mkdir(parents=True, exist_ok=True)
        Path(args.blob_store).parent.mkdir(parents=True, exist_ok=True)

        # Fix empty/corrupt JSONStore files before connecting
        for store_path in (Path(args.docs_store), Path(args.blob_store)):
            if store_path.exists() and store_path.stat().st_size == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: {store_path} is empty; reinitializing it as []")
                store_path.write_text("[]", encoding="utf-8")

        docs_store = JSONStore(args.docs_store, read_only=False)
        blob_store = JSONStore(args.blob_store, read_only=False)
        store = JobStore(docs_store, additional_stores={"data": blob_store})












        # ── Step 4: Run workflow locally ──────────────────────────────────
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

        # ── Step 5: Timing summary + completion message ───────────────────
        timing_handler.print_summary(args.mpid)

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] ✔ Workflow completed successfully!")
        print(f"  Results  → {args.docs_store}")
        print(f"  Blob     → {args.blob_store}")
        print(f"  Full log → {out_path}")
        print(f"  Err  log → {err_path}")

    except Exception:
        ts = datetime.now().strftime("%H:%M:%S")
        tb = traceback.format_exc()
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
