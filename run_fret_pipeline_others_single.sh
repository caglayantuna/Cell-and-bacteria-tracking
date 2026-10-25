#!/bin/bash
#SBATCH --job-name=fret_others
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/pasteur/helix/users/egokerku/results/fret_others_%j.out
#SBATCH --error=/pasteur/helix/users/egokerku/results/fret_others_%j.err

# ── Single-control run (the normal case): all wells use ONE control ──
# No --wells filter → every well in the dataset is processed against --control_well.
DATASET="MY_DATASET"                 # measurement folder name, or an absolute path
CONTROL_WELL="r02c05"                # the one control well for delta-R
EXCLUSION="edge"

source /pasteur/helix/users/egokerku/fret_env/bin/activate

python /pasteur/helix/users/egokerku/begum_maestro_codes/run_fret_pipeline_others.py \
    "${DATASET}" \
    --control_well "${CONTROL_WELL}" \
    --exclusion "${EXCLUSION}"
