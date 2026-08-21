#!/usr/bin/env bash
# Turbomole env bootstrap for int-nano (to be SOURCED, not executed).
# Keep this file side-effect free: do not run define/ridft/escf here.

# Initialize environment-modules in non-interactive shells if needed
if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/modules.sh
  elif [ -f /usr/share/Modules/init/bash ]; then
    # shellcheck disable=SC1091
    . /usr/share/Modules/init/bash
  elif [ -f /usr/share/module/init/bash ]; then
    # shellcheck disable=SC1091
    . /usr/share/module/init/bash
  fi
fi

# Load Turbomole
module purge >/dev/null 2>&1 || true
module load turbomole/7.6

# Optional: threads
#export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
