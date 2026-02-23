#!/bin/bash
#SBATCH --job-name=pheasy_fit_mp556595
#SBATCH --account=matgen
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ---- user settings ----
WORKDIR="/pscratch/sd/h/hrushi99/atomate2/pheasy/nesap/fom_1/mp-556595/rank_5_v2"
ENVNAME="atomate2_pheasy"
# -----------------------

echo "Job started on $(hostname) at $(date)"
echo "WORKDIR: ${WORKDIR}"
cd "${WORKDIR}"

# Activate conda env
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found in PATH and conda.sh not found in common locations."
    exit 1
  fi
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate "${ENVNAME}"

# Threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Conda env: $(which python)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

run_cmd () {
  local label="$1"
  shift
  echo
  echo "==== ${label} ===="
  echo "CMD: $*"
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" "$@" 2>&1 | tee "${label}.log"
}

run_cmd "01_pheasy_s" pheasy -s --dim 3 3 3 -w 2 --nbody 2
run_cmd "02_pheasy_c" pheasy -c --dim 3 3 3 -w 2
run_cmd "03_pheasy_d" pheasy -d --dim 3 3 3 --ndata 12 --disp_file
run_cmd "04_pheasy_f" pheasy -f --dim 3 3 3 --ndata 12 -w 2 -l LASSO --std --full_ifc --rasr BHH --hdf5
run_cmd "05_phonopy"  phonopy --dim 3 3 3 --readfc --band auto -ps --nac

echo
echo "Job finished at $(date)"
