#!/bin/bash
#SBATCH --job-name=fret_pipeline
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/egokerku/results/fret_pipeline_%j.out
#SBATCH --error=/pasteur/helix/users/egokerku/results/fret_pipeline_%j.err

# ── Edit these per run ──
DATASET="260304_NCI_40xW_timecourse_mcherrybacteria__2026-03-04T15_09_07-Measurement"
CONTROL_FIELD="none"
EXCLUSION="edge"

source /pasteur/helix/users/egokerku/fret_env/bin/activate

python /pasteur/helix/users/egokerku/begum_maestro_codes/run_fret_pipeline.py "${DATASET}" --control_field "${CONTROL_FIELD}" --exclusion "${EXCLUSION}"
