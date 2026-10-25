#!/bin/bash
#SBATCH --job-name=fret_pipeline_multi
#SBATCH --array=0-1
#SBATCH --time=30:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/egokerku/results/fret_pipeline_multi_%A_%a.out
#SBATCH --error=/pasteur/helix/users/egokerku/results/fret_pipeline_multi_%A_%a.err

# ── Multi-control run: 2+ controls, each with its own column of data wells,
#    all living in the SAME dataset directory. One array task per control;
#    --wells restricts each task to its own column so all tasks can safely
#    share one save_dir. Add more entries + bump #SBATCH --array to match if
#    you have more than 2 controls (e.g. --array=0-2 for 3, as in the
#    "others" pipeline's run_fret_pipeline_others.sh).
DATASET="260304_NCI_40xW_timecourse_mcherrybacteria__2026-03-04T15_09_07-Measurement"
EXCLUSION="edge"

# One entry per array task. CTRL is used for BOTH --control_well (delta-R
# baseline) and --control_field (no-bacteria field, skips OmniPose + contact
# detection there) -- the same well, for both purposes. Control r02cXX pairs
# with column cXX (r02cXX..r07cXX).
CONTROLS=(r02c05 r02c06)
COLUMNS=(c05    c06)

CTRL=${CONTROLS[$SLURM_ARRAY_TASK_ID]}
COL=${COLUMNS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID}: control=${CTRL}  column=${COL}"

source /pasteur/helix/users/egokerku/fret_env/bin/activate

python /pasteur/helix/users/egokerku/begum_maestro_codes/run_fret_pipeline.py \
    "${DATASET}" \
    --control_well "${CTRL}" \
    --control_field "${CTRL}" \
    --wells "${COL}" \
    --exclusion "${EXCLUSION}"
