# FRET bacteria-cell pipeline (Maestro cluster)

Segments cells (CellPose) and bacteria (OmniPose), measures the ch2/ch1 FRET ratio per cell over time, and computes delta-R against a control well.

## Scripts

| Script | Use for |
|---|---|
| `run_fret_pipeline.py` | Data with a bacteria channel (ch3). Adds bacteria-contact detection. |
| `run_fret_pipeline_others.py` | Data without ch3. Bacteria are segmented on ch1. No contact analysis. Adds a cell area table. |
| `*_perpixel.py` | Same as above, but the ratio is the mean of per-pixel ch2/ch1 (median filter + ch1 > 3 x background SD). Use these going forward. |
| `run_ratio_tiffs.py` | Saves the per-pixel ratio as TIFF images (plus outline composites for Fiji). |
| `compare_ratio_methods.py` | Compares ratio of means with mean of per-pixel ratios on finished results. |

## Launchers (SLURM)

| File | Use for |
|---|---|
| `run_fret_pipeline.sh` | One control for the whole plate. |
| `run_fret_pipeline_multi_control.sh` | Several controls in one dataset. One array task per control column. |
| `run_fret_pipeline_others.sh` | Same, for the "others" script (3 controls). |
| `run_fret_pipeline_others_single.sh` | "Others" script with a single control. |

The launchers still call the non-perpixel scripts. Change the script name to use a `_perpixel` version.

## Run

```bash
sbatch run_fret_pipeline_others_single.sh
```

Edit `DATASET`, `CONTROL_WELL` and `EXCLUSION` at the top of the launcher first. Runs are resumable.

## Output

Per field: `results_ratio.csv`, `results_c1.csv`, `results_c2.csv`, `cell_seg/overlays/`, `bacteria_seg/overlays/`.
Per well: `ratio_all_field.csv`, `deltar_fret.csv`. Control well also gets `control_population_mean.csv`.

`some_results/` has example overlay videos.
