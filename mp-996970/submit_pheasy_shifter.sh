#!/bin/bash
#SBATCH --job-name=pheasy_fit_mp996970
#SBATCH --account=matgen
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

IMAGE="ghcr.io/hrushikesh-s/pheasy-phonopy:phonopy2.28.0-pheasy0.0.2-py3.9-numpy1.26"
WORKDIR="/pscratch/sd/h/hrushi99/atomate2/pheasy/nesap/fom_1/mp-996970/mp-996970_new_supercell_iteration1_displaced_configs/965814"

cd "$WORKDIR"

# run inside container
shifter --image="$IMAGE" bash -lc "./run_pheasy_phonopy_container.sh"
