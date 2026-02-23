#!/bin/bash
export PATH=/opt/conda/envs/atomate2_pheasy/bin:$PATH

# Threads (Slurm passes these env vars into the container)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Inside container:"
which python
python -V
python -c "import numpy as np; print('numpy', np.__version__)"
pheasy --version
python -c "import phonopy; print('phonopy', phonopy.__version__)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

run_cmd () {
  local label="$1"; shift
  echo
  echo "==== ${label} ===="
  echo "CMD: $*"
  "$@" 2>&1 | tee "${label}.log"
}

run_cmd "01_pheasy_s" pheasy -s --dim 3 3 3 -w 2 --nbody 2
run_cmd "02_pheasy_c" pheasy -c --dim 3 3 3 -w 2
run_cmd "03_pheasy_d" pheasy -d --dim 3 3 3 --ndata 18 --disp_file
run_cmd "04_pheasy_f" pheasy -f --dim 3 3 3 --ndata 18 -w 2 -l LASSO --std --full_ifc --rasr BHH --hdf5
run_cmd "05_phonopy"  phonopy --dim 3 3 3 --readfc --band auto -ps --nac

echo "Done at $(date)"
