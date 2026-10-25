"""
compare_ratio_methods.py

Standalone comparison script — does NOT modify run_fret_pipeline.py /
run_fret_pipeline_others.py or any of their outputs.

For every (well, field, timepoint, cell) already analysed by the pipeline,
recomputes the per-cell ratio TWO ways from the same background-corrected
ch1/ch2 images and the same cell masks already on disk:

    ratio_of_means   = mean(ch2_corr[cell_px]) / mean(ch1_corr[cell_px])
                        <- what run_fret_pipeline*.py currently writes to
                           results_ratio.csv (mean_ratio column)

    mean_of_ratios   = mean( ch2_corr[px] / ch1_corr[px]  for px in cell_px )
                        <- what the researcher asked for: the per-pixel
                           ratio, averaged over the cell's pixels.
                           Pixels with ch1_corr == 0 (background-clipped,
                           undefined ratio) are excluded and counted.

Writes one CSV per field:
    <save_dir>/<well>/<field>/ratio_method_comparison.csv
with columns: time, cell_id, ratio_of_means, mean_of_ratios,
              pct_diff, n_pixels, n_zero_denom_px

and prints an overall summary (correlation + median/mean relative
difference) across every cell/timepoint processed.

Usage (mirrors run_fret_pipeline.py's directory resolution):
    python compare_ratio_methods.py <dataset> \
        --base_dir /pasteur/helix/scratch/egokerku/FRET/ \
        --results_dir /pasteur/helix/users/egokerku/results/ \
        [--output_name NAME] [--wells r02c05,r02c06]
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from skimage.io import imread

DEFAULT_BASE_DIR    = "/pasteur/helix/scratch/egokerku/FRET/"
DEFAULT_RESULTS_DIR = "/pasteur/helix/users/egokerku/results/"


def parse_filename(fname):
    well  = re.search(r'r\d+c\d+', fname).group()
    field = re.search(r'f\d+', fname).group()
    p     = int(re.search(r'p(\d+)', fname).group(1))
    ch    = re.search(r'ch\d+', fname).group()
    sk    = int(re.search(r'sk(\d+)', fname).group(1))
    return well, field, p, ch, sk


def group_files(directory):
    """Organise TIFF files into {(well, field): {timepoint: {plane: {channel: filename}}}}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in os.listdir(directory):
        if not (f.endswith(".tiff") or f.endswith(".tif")):
            continue
        well, field, p, ch, sk = parse_filename(f)
        data[(well, field)][sk][p][ch] = f
    return {
        key: {
            sk: dict(sorted(planes.items()))
            for sk, planes in sorted(times.items())
        }
        for key, times in sorted(data.items())
    }


def _filter_wells(pos_dict, wells_arg):
    tokens = [w.strip() for w in (wells_arg or "").split(",") if w.strip()]
    if not tokens or "all" in tokens:
        return pos_dict
    def match(well):
        for tok in tokens:
            if re.fullmatch(r"r\d+c\d+", tok) and well == tok:
                return True
            if re.fullmatch(r"c\d+", tok) and re.fullmatch(rf"r\d+{tok}", well):
                return True
        return False
    return {k: v for k, v in pos_dict.items() if match(k[0])}


def compare_field(well, field, time_dict, directory, save_dir):
    field_dir = os.path.join(save_dir, well, field)
    mask_dir      = os.path.join(field_dir, "cell_seg", "masks")
    mask_full_dir = os.path.join(field_dir, "cell_seg", "masks_full")

    if not os.path.isdir(mask_dir):
        print(f"  [{well}/{field}] no cell_seg/masks/ found — skipped "
              f"(masks likely deleted; re-run with --keep_masks to compare).")
        return None

    rows = []
    for t, planes in time_dict.items():
        mask_path = os.path.join(mask_dir, f"mask_sk{t}.npy")
        if not os.path.exists(mask_path):
            continue
        cell_mask = np.load(mask_path)

        full_mask_path = os.path.join(mask_full_dir, f"mask_sk{t}.npy")
        full_cell_mask = (np.load(full_mask_path)
                           if os.path.exists(full_mask_path) else cell_mask)

        if 2 not in planes or "ch1" not in planes[2] or "ch2" not in planes[2]:
            continue
        ch1_raw = imread(os.path.join(directory, planes[2]['ch1'])).astype(float)
        ch2_raw = imread(os.path.join(directory, planes[2]['ch2'])).astype(float)

        # Same background correction as run_analysis_pass: background excludes
        # both cell footprints and bacteria (when a bacteria mask exists for
        # this field/timepoint; control fields have none -> treated as zeros,
        # matching is_control in the pipeline).
        bact_mask_path = os.path.join(field_dir, "bacteria_seg", "masks", f"mask_sk{t}.npy")
        if os.path.exists(bact_mask_path):
            bact_mask = np.load(bact_mask_path) > 0
        else:
            bact_mask = np.zeros(cell_mask.shape, dtype=bool)

        bg_mask = (full_cell_mask == 0) & ~bact_mask
        if bg_mask.any():
            bg_ch1 = float(ch1_raw[bg_mask].mean())
            bg_ch2 = float(ch2_raw[bg_mask].mean())
        else:
            bg_ch1 = float(np.percentile(ch1_raw, 5))
            bg_ch2 = float(np.percentile(ch2_raw, 5))

        ch1_corr = np.clip(ch1_raw - bg_ch1, 0, None)
        ch2_corr = np.clip(ch2_raw - bg_ch2, 0, None)

        for cell_id in np.unique(cell_mask):
            if cell_id == 0:
                continue
            px = cell_mask == cell_id
            c1_px = ch1_corr[px]
            c2_px = ch2_corr[px]

            mc1 = float(c1_px.mean())
            mc2 = float(c2_px.mean())
            ratio_of_means = mc2 / mc1 if mc1 > 0 else np.nan

            denom_ok = c1_px > 0
            n_zero_denom = int((~denom_ok).sum())
            if denom_ok.any():
                mean_of_ratios = float((c2_px[denom_ok] / c1_px[denom_ok]).mean())
            else:
                mean_of_ratios = np.nan

            pct_diff = (
                100.0 * (mean_of_ratios - ratio_of_means) / ratio_of_means
                if ratio_of_means not in (0, np.nan) and not np.isnan(ratio_of_means)
                else np.nan
            )

            rows.append({
                "time":            t,
                "cell_id":         int(cell_id),
                "ratio_of_means":  round(ratio_of_means, 4),
                "mean_of_ratios":  round(mean_of_ratios, 4) if not np.isnan(mean_of_ratios) else np.nan,
                "pct_diff":        round(pct_diff, 2) if not np.isnan(pct_diff) else np.nan,
                "n_pixels":        int(px.sum()),
                "n_zero_denom_px": n_zero_denom,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values(["time", "cell_id"])
    out_path = os.path.join(field_dir, "ratio_method_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"  [{well}/{field}] wrote {out_path}  ({len(df)} cell-timepoints)")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare ratio-of-means vs mean-of-ratios per cell, "
                     "without touching the pipeline's own outputs.")
    parser.add_argument("dataset",
                        help="Measurement folder name inside --base_dir "
                             "(or an absolute path to a folder of TIFFs).")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_name", default=None,
                        help="Result subfolder name (default: same as dataset).")
    parser.add_argument("--wells", default="all")
    args = parser.parse_args()

    if os.path.isabs(args.dataset):
        directory = args.dataset
        dataset_name = os.path.basename(os.path.normpath(args.dataset))
    else:
        directory = os.path.join(args.base_dir, args.dataset)
        dataset_name = args.dataset

    if not os.path.isdir(directory):
        raise SystemExit(f"Input directory does not exist: {directory}")

    for images_subdir in ("Images", "images"):
        candidate = os.path.join(directory, images_subdir)
        if os.path.isdir(candidate):
            directory = candidate
            break

    output_name = args.output_name or dataset_name
    save_dir = os.path.join(args.results_dir, output_name)

    pos_dict = group_files(directory)
    pos_dict = _filter_wells(pos_dict, args.wells)

    all_dfs = []
    for (well, field), time_dict in pos_dict.items():
        df = compare_field(well, field, time_dict, directory, save_dir)
        if df is not None:
            df["well"] = well
            df["field"] = field
            all_dfs.append(df)

    if not all_dfs:
        print("No comparable data found (check --wells / masks were kept).")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    valid = combined.dropna(subset=["ratio_of_means", "mean_of_ratios"])

    print("\n===== Summary across all cell-timepoints =====")
    print(f"n = {len(valid)}")
    print(f"Pearson correlation (ratio_of_means vs mean_of_ratios): "
          f"{valid['ratio_of_means'].corr(valid['mean_of_ratios']):.4f}")
    print(f"Median % difference: {valid['pct_diff'].median():.2f}%")
    print(f"Mean % difference:   {valid['pct_diff'].mean():.2f}%")
    print(f"Mean abs % difference: {valid['pct_diff'].abs().mean():.2f}%")
    print(f"Total zero-denominator pixels skipped: {int(combined['n_zero_denom_px'].sum())} "
          f"/ {int(combined['n_pixels'].sum())} total cell-pixels")


if __name__ == "__main__":
    main()
