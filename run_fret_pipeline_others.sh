#!/bin/bash
#SBATCH --job-name=fret_others
#SBATCH --array=0-2
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/egokerku/results/fret_others_%A_%a.out
#SBATCH --error=/pasteur/helix/users/egokerku/results/fret_others_%A_%a.err

# ── Edit per run ──
DATASET="/pasteur/helix/scratch/egokerku/FRET/260320_NCI_40xW_timecourse_exoYvariants_rep1_2_3__2026-03-20T14_53_33-Measurement 1"         # measurement folder under --base_dir (all 18 wells live here)
EXCLUSION="edge"

# One entry per array task. Control r02cXX pairs with column cXX (r02cXX..r07cXX).
CONTROLS=(r02c05 r02c06 r02c07)
COLUMNS=(c05    c06    c07)

CTRL=${CONTROLS[$SLURM_ARRAY_TASK_ID]}
COL=${COLUMNS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID}: control=${CTRL}  column=${COL}"

source /pasteur/helix/users/egokerku/fret_env/bin/activate

python /pasteur/helix/users/egokerku/begum_maestro_codes/run_fret_pipeline_others.py \
    "${DATASET}" \
    --control_well "${CTRL}" \
    --wells "${COL}" \
    --exclusion "${EXCLUSION}"
