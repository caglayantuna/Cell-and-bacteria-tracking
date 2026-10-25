"""
run_fret_pipeline.py

Four-pass pipeline for FRET bacteria-cell interaction analysis, reorganised to
run on the Maestro / helix SLURM cluster.

This is the same validated pipeline as run_tracking_with_bacteria.py, with the
entry point rewired for the cluster's directory layout:

    data   : <base_dir>/<dataset>/Images/                (contains the TIFF files;
                                                          falls back to <base_dir>/<dataset>/
                                                          directly if no Images/images subfolder exists)
    results: <results_dir>/<dataset>/  (or <results_dir>/<output_name>/)

Defaults for this cluster:
    base_dir    = /pasteur/helix/scratch/egokerku/FRET/
    results_dir = /pasteur/helix/users/egokerku/results/

Videos are NOT generated (Pass 5 only regenerates the still overlays); pass
--no_postprocess to skip Pass 5 entirely.

Pipeline
--------
Pass 1  Bacteria segmentation (OmniPose, ch3). For each timepoint, all
        available planes are max-intensity-projected into a single image
        (then CLAHE-preprocessed) and segmented with ONE OmniPose call.
        Skipped for control field(s) with no bacteria channel signal.
Pass 2  Cell segmentation + frame-to-frame tracking (CellPose, ch2 @ plane 2,
        CLAHE-preprocessed, diameter=100). Optionally excludes border cells
        and/or dying/dividing cells per --exclusion {none, edge, both}.
Pass 3  Analysis (CPU only): background-corrected ch1/ch2 means per cell per
        frame, ch2/ch1 ratio, bacteria-contact detection -> CSVs (no overlays
        written here; see Pass 5).
Pass 4  delta-R FRET reports: ratio_all_field.csv / control_population_mean.csv
        / deltar_fret.csv. Needs --control_well; skipped if not given.
Pass 5  Post-processing: (re)creates cell_seg/overlays/ and bacteria_seg/overlays/
        (each only once, skip if present) with a deterministic per-cell palette,
        white borders, and ID labels. No videos. Skip with --no_postprocess.
        Then mask directories are removed to save space, unless --keep_masks.

Result layout
-------------
    <results_dir>/<output_name>/<well>/<field>/bacteria_seg/masks/mask_sk<t>.npy
    <results_dir>/<output_name>/<well>/<field>/bacteria_seg/overlays/overlay_sk<t>.png
    <results_dir>/<output_name>/<well>/<field>/cell_seg/masks/mask_sk<t>.npy
    <results_dir>/<output_name>/<well>/<field>/cell_seg/overlays/overlay_sk<t>.png
    <results_dir>/<output_name>/<well>/<field>/images/image_sk<t>.png
    <results_dir>/<output_name>/<well>/<field>/results_ratio.csv
    <results_dir>/<output_name>/<well>/<field>/results_c1.csv
    <results_dir>/<output_name>/<well>/<field>/results_c2.csv
    <results_dir>/<output_name>/<well>/<field>/contact_matrix.csv
    <results_dir>/<output_name>/<well>/<field>/excluded_cells.csv

All passes are resumable: each checks the output for previously written outputs
and skips work that is already done.

Usage
-----
    python run_fret_pipeline.py <dataset> [options]

Positional argument
    dataset         Name of the measurement folder inside --base_dir, e.g.
                    260304_NCI_40xW_timecourse_mcherrybacteria__2026-03-04T15_09_07-Measurement
                    (or an absolute path to a folder of TIFFs).

Optional arguments
    --base_dir DIR              Root folder that contains <dataset>
                                (default: /pasteur/helix/scratch/egokerku/FRET/).
    --results_dir DIR           Root folder for results
                                (default: /pasteur/helix/users/egokerku/results/).
    --output_name NAME          Result subfolder name (default: same as dataset).
    --control_field TOKENS      No-bacteria control(s). A whole well (e.g. r02c05)
                                makes ALL its fields control; a well+field token
                                (e.g. r02c05f01) makes only that field control.
                                Comma-separate for several; 'none' if no control.
    --exclusion {none,edge,both}  Cell exclusion policy (default: edge).
                                  none = keep every cell
                                  edge = drop cells touching the image border
                                  both = drop border cells AND dying/dividing cells
    --proximity_px INT          Pixel dilation radius for contact detection (default: 3).
    --flow_threshold FLOAT      CellPose flow threshold (default: 0.4).
    --cellprob_threshold FLOAT  CellPose cell probability threshold (default: 0).
    --area_change_threshold FLOAT  Frame-to-frame area ratio bound for 'both' (default: 0.5).
    --no_postprocess            Skip Pass 4 (overlay regeneration).

Examples
    # edge exclusion (the default), no control field
    python run_fret_pipeline.py 260304_NCI_40xW_timecourse_mcherrybacteria__2026-03-04T15_09_07-Measurement

    # whole well r02c05 (all its fields) is the no-bacteria control
    python run_fret_pipeline.py my_dataset --control_field r02c05

    # a single control field, plus dying/dividing exclusion
    python run_fret_pipeline.py my_dataset --control_field r02c05f01 --exclusion both
"""

import os
import re
import gc
import glob
import shutil
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.exposure import equalize_adapthist
from skimage.segmentation import find_boundaries
from collections import defaultdict

import cellpose
import cellpose_omni
from cellpose_omni import models
from omnipose.gpu import use_gpu
from cellpose.models import CellposeModel

from matplotlib import cm
from scipy.ndimage import center_of_mass, binary_dilation
import torch


EXCLUSION_MODES = ("none", "edge", "both")

# Cluster defaults
DEFAULT_BASE_DIR    = "/pasteur/helix/scratch/egokerku/FRET/"
DEFAULT_RESULTS_DIR = "/pasteur/helix/scratch/egokerku/results/"

# Fixed palette for post-processed overlays: cell_id % N always maps to the
# same color, so a given cell keeps its color across every frame.
_OVERLAY_PALETTE = np.array([
    [0.90, 0.20, 0.20],  # red
    [0.20, 0.60, 0.90],  # blue
    [0.20, 0.80, 0.30],  # green
    [0.95, 0.70, 0.10],  # orange
    [0.70, 0.30, 0.90],  # purple
    [0.10, 0.80, 0.80],  # cyan
    [0.95, 0.40, 0.70],  # pink
    [0.50, 0.85, 0.20],  # lime
    [0.90, 0.55, 0.20],  # amber
    [0.30, 0.50, 0.90],  # indigo
    [0.80, 0.20, 0.50],  # magenta
    [0.20, 0.70, 0.55],  # teal
    [0.95, 0.85, 0.20],  # yellow
    [0.60, 0.40, 0.20],  # brown
    [0.55, 0.55, 0.90],  # lavender
    [0.20, 0.40, 0.60],  # navy
    [0.85, 0.50, 0.50],  # salmon
    [0.40, 0.80, 0.60],  # mint
    [0.80, 0.65, 0.30],  # gold
    [0.50, 0.20, 0.70],  # violet
])


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def format_duration(seconds):
    """Format a duration in seconds as 'Hh Mm Ss' (e.g. '1h 03m 47s')."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def save_overlay(image, masks, save_path, write_id=True):
    """Save a PNG overlay of a grayscale image with coloured segmentation masks."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    if masks.max() > 0:
        colored_masks = cm.tab20(masks / (masks.max() + 1e-6))
        ax.imshow(colored_masks, alpha=0.4)
        if write_id:
            for cell_id in np.unique(masks):
                if cell_id == 0:
                    continue
                cy, cx = center_of_mass(masks == cell_id)
                ax.text(cx, cy, str(cell_id), color="white", fontsize=8,
                        ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    del fig, ax


def parse_filename(fname):
    """Extract well, field, focal-plane index, channel, and timepoint from a filename."""
    well  = re.search(r'r\d+c\d+', fname).group()
    field = re.search(r'f\d+', fname).group()
    p     = int(re.search(r'p(\d+)', fname).group(1))
    ch    = re.search(r'ch\d+', fname).group()
    sk    = int(re.search(r'sk(\d+)', fname).group(1))
    return well, field, p, ch, sk


def resolve_control_fields(tokens, all_keys):
    """Expand control tokens into a set of concrete (well, field) pairs.

    A token may be either:
      - a whole well, e.g. 'r02c05'      -> ALL fields of that well become control
      - a single field, e.g. 'r02c05f01' -> only that one field becomes control

    *all_keys* is the collection of (well, field) pairs actually present in the
    dataset (i.e. pos_dict.keys()). Tokens that match nothing are reported.
    """
    present = set(all_keys)
    control = set()
    for raw in tokens:
        token = raw.strip()
        m_field = re.match(r'(r\d+c\d+)(f\d+)$', token)
        m_well  = re.match(r'(r\d+c\d+)$', token)
        if m_field:
            key = (m_field.group(1), m_field.group(2))
            if key in present:
                control.add(key)
            else:
                print(f"  [warn] control field {token!r} not found in dataset — ignored")
        elif m_well:
            well = m_well.group(1)
            matched = {(w, f) for (w, f) in present if w == well}
            if matched:
                control |= matched
                print(f"  [control] well {well} → all {len(matched)} field(s): "
                      f"{sorted(f for _, f in matched)}")
            else:
                print(f"  [warn] control well {token!r} not found in dataset — ignored")
        else:
            raise ValueError(
                f"Expected a well token like 'r02c05' or a well+field token like "
                f"'r02c05f01', got {token!r}")
    return control


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


def max_project_channel(directory, planes, plane_ids, channel):
    """Max-intensity projection of a raw channel across the given planes."""
    stack = np.stack([
        imread(os.path.join(directory, planes[p][channel])).astype(float)
        for p in plane_ids
    ], axis=0)
    return stack.max(axis=0)


def track_cells(prev_masks, curr_masks):
    """Propagate cell IDs across frames using majority pixel-overlap matching."""
    new_masks = np.zeros_like(curr_masks)
    next_id   = int(prev_masks.max()) + 1
    for curr_id in np.unique(curr_masks):
        if curr_id == 0:
            continue
        region      = curr_masks == curr_id
        overlapping = prev_masks[region]
        overlapping = overlapping[overlapping > 0]
        if overlapping.size > 0:
            new_masks[region] = int(np.bincount(overlapping).argmax())
        else:
            new_masks[region] = next_id
            next_id += 1
    return new_masks


def get_edge_cells(masks_dict):
    """Return the set of cell IDs that ever touch the image border, across all frames."""
    edge_ids = set()
    for masks in masks_dict.values():
        border = np.zeros_like(masks, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        ids = set(np.unique(masks[border]))
        ids.discard(0)
        edge_ids.update(ids)
    return edge_ids


def get_problematic_cells(masks_dict, area_change_threshold=0.5):
    """Flag cells that disappear before the last timepoint or whose area jumps
    abruptly between consecutive frames (candidate dying/dividing cells)."""
    timepoints = sorted(masks_dict.keys())
    last_t     = timepoints[-1]
    all_ids = set()
    for m in masks_dict.values():
        all_ids.update(np.unique(m))
    all_ids.discard(0)
    areas = {cid: {} for cid in all_ids}
    for t, masks in masks_dict.items():
        for cid in np.unique(masks):
            if cid == 0:
                continue
            areas[cid][t] = int(np.sum(masks == cid))
    flagged = set()
    for cid, ts in areas.items():
        if not ts:
            continue
        present = sorted(ts.keys())
        if present[-1] < last_t:
            flagged.add(cid)
            continue
        for i in range(1, len(present)):
            ratio = ts[present[i]] / ts[present[i - 1]]
            if ratio < area_change_threshold or ratio > 1.0 / area_change_threshold:
                flagged.add(cid)
                break
    return flagged


def remove_cells_from_masks(masks_dict, cell_ids):
    """Zero out the given cell IDs from every frame's mask."""
    cleaned = {}
    for t, masks in masks_dict.items():
        m = masks.copy()
        for cid in cell_ids:
            m[m == cid] = 0
        cleaned[t] = m
    return cleaned


def detect_interactions(masks_cell, bact_mask, proximity_px=3):
    """Return cell IDs whose footprint is within *proximity_px* pixels of any bacterium."""
    if not bact_mask.any():
        return set()
    struct  = np.ones((2 * proximity_px + 1, 2 * proximity_px + 1), dtype=bool)
    dilated = binary_dilation(bact_mask, structure=struct)
    return {
        int(cid)
        for cid in np.unique(masks_cell)
        if cid != 0 and np.logical_and(masks_cell == cid, dilated).any()
    }


def save_wide(df, value_col, filename, save_dir):
    """Pivot long-format DataFrame to wide format (rows=time, cols=cells) and save CSV."""
    df_wide = df.pivot(index="time", columns="cell_id", values=value_col)
    df_wide.columns = [f"cell_{c}" for c in df_wide.columns]
    df_wide.sort_index(inplace=True)
    df_wide.to_csv(os.path.join(save_dir, filename))


def _save_exclusion_log(edge_cells, bad_cells, field_dir, tag):
    rows = (
        [{"cell_id": cid, "reason": "edge"} for cid in sorted(edge_cells)]
        + [{"cell_id": cid, "reason": "dying_or_dividing"}
           for cid in sorted(bad_cells - edge_cells)]
    )
    if rows:
        path = os.path.join(field_dir, "excluded_cells.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"    [{tag}] Excluded {len(rows)} cells → {path}")
    else:
        print(f"    [{tag}] No cells excluded.")


# ── Resume helpers ────────────────────────────
def _omnipose_frame_done(save_dir, well, field, t):
    # Done if the mask OR the overlay exists (masks may be cleaned up after the
    # overlay is written); only re-segment when BOTH are missing.
    bseg = os.path.join(save_dir, well, field, "bacteria_seg")
    return (os.path.exists(os.path.join(bseg, "masks", f"mask_sk{t}.npy"))
            or os.path.exists(os.path.join(bseg, "overlays", f"overlay_sk{t}.png")))

def _cellpose_field_done(save_dir, well, field, time_dict):
    # Done if analysis already ran (masks may have been cleaned up to save space).
    if _analysis_field_done(save_dir, well, field):
        return True
    mask_dir = os.path.join(save_dir, well, field, "cell_seg", "masks")
    return all(
        os.path.exists(os.path.join(mask_dir, f"mask_sk{t}.npy"))
        for t in time_dict
    )

def _analysis_field_done(save_dir, well, field):
    return os.path.exists(os.path.join(save_dir, well, field, "results_ratio.csv"))


# ──────────────────────────────────────────────
# Pass 1 — Bacteria segmentation (OmniPose)
# ──────────────────────────────────────────────

def run_omnipose_pass(pos_dict, directory, save_dir, params_omnipose,
                      no_bacteria_fields=None):
    no_bact = set(no_bacteria_fields or [])

    todo = [
        (key, t)
        for key, time_dict in pos_dict.items()
        if key not in no_bact
        for t in time_dict
        if not _omnipose_frame_done(save_dir, key[0], key[1], t)
    ]
    if not todo:
        skipped = f"; control fields skipped: {sorted(no_bact)}" if no_bact else ""
        print(f"OmniPose pass — all timepoints already done, skipping{skipped}.\n")
        return
    print(f"OmniPose pass — {len(todo)} timepoint(s) remaining.")

    use_GPU = use_gpu()
    model   = models.CellposeModel(gpu=use_GPU, model_type='bact_fluor_omni')

    for (well, field), time_dict in pos_dict.items():
        if (well, field) in no_bact:
            print(f"  [OmniPose] {well}/{field} — skipped (no-bacteria control)")
            continue
        if all(_omnipose_frame_done(save_dir, well, field, t) for t in time_dict):
            print(f"  [OmniPose] {well}/{field} — already done, skipping.")
            continue
        planes_available = sorted(next(iter(time_dict.values())).keys())
        print(f"  [OmniPose] {well}/{field}  max-projecting planes={planes_available}  "
              f"({len(time_dict)} timepoints)")

        mask_dir = os.path.join(save_dir, well, field, "bacteria_seg", "masks")
        os.makedirs(mask_dir, exist_ok=True)

        for t, planes in time_dict.items():
            if _omnipose_frame_done(save_dir, well, field, t):
                continue
            raw = max_project_channel(directory, planes, planes_available, 'ch3')
            img = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
            del raw
            seg_img = equalize_adapthist(img, clip_limit=0.03)
            del img

            masks, _, _ = model.eval(seg_img, **params_omnipose)
            np.save(os.path.join(mask_dir, f"mask_sk{t}.npy"),
                    masks.astype(np.uint16))
            # Bacteria overlays are written in Pass 5 (like the cell overlays), not here.
            del seg_img, masks
            torch.cuda.empty_cache()
            gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("OmniPose pass complete — model unloaded.\n")


# ──────────────────────────────────────────────
# Pass 2 — Cell segmentation + tracking (CellPose)
# ──────────────────────────────────────────────

def run_cellpose_pass(pos_dict, directory, save_dir,
                      flow_threshold=0.4, cellprob_threshold=0,
                      area_change_threshold=0.5, exclusion="both"):
    """Segment + track cells, then apply one of three exclusion policies:
        "none" - keep every tracked cell
        "edge" - drop cells that ever touch the image border
        "both" - drop border cells AND cells with abrupt area changes /
                 early disappearance (dying/dividing candidates)
    """
    if exclusion not in EXCLUSION_MODES:
        raise ValueError(f"exclusion must be one of {EXCLUSION_MODES}, got {exclusion!r}")

    fields_todo = []
    for key, time_dict in pos_dict.items():
        if _cellpose_field_done(save_dir, key[0], key[1], time_dict):
            print(f"  [CellPose] {key[0]}/{key[1]} — already done, skipping.")
        else:
            fields_todo.append(key)
    if not fields_todo:
        print("CellPose pass — all fields already done, skipping.\n")
        return
    print(f"CellPose pass (exclusion={exclusion}) — processing: {fields_todo}")

    model = CellposeModel(gpu=True)

    for well, field in fields_todo:
        time_dict = pos_dict[(well, field)]
        mask_dir      = os.path.join(save_dir, well, field, "cell_seg", "masks")
        mask_full_dir = os.path.join(save_dir, well, field, "cell_seg", "masks_full")
        os.makedirs(mask_dir,      exist_ok=True)
        os.makedirs(mask_full_dir, exist_ok=True)
        print(f"  [CellPose] {well}/{field}  ({len(time_dict)} timepoints)")

        masks_all  = {}
        prev_masks = None
        for t, planes in time_dict.items():
            ch2     = imread(os.path.join(directory, planes[2]['ch2'])).astype(float)
            img     = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)
            del ch2
            seg_img = equalize_adapthist(img, clip_limit=0.03)
            del img

            masks, _, _ = model.eval(
                seg_img,
                batch_size=8,
                diameter=100,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=False,
            )
            del seg_img
            torch.cuda.empty_cache()
            gc.collect()

            if prev_masks is not None:
                masks = track_cells(prev_masks, masks)
            prev_masks   = masks.copy()
            masks_all[t] = masks
            del masks
            gc.collect()
        del prev_masks

        if exclusion == "none":
            edge_cells, bad_cells = set(), set()
        elif exclusion == "edge":
            edge_cells = get_edge_cells(masks_all)
            bad_cells  = set()
        else:  # "both"
            edge_cells = get_edge_cells(masks_all)
            bad_cells  = get_problematic_cells(masks_all, area_change_threshold)

        exclude = edge_cells | bad_cells
        cleaned = remove_cells_from_masks(masks_all, exclude) if exclude else masks_all

        _save_exclusion_log(edge_cells, bad_cells,
                            os.path.join(save_dir, well, field), f"{well}/{field}")

        # Full (pre-exclusion) masks: used ONLY to define the background in
        # Pass 3, so that excluded edge/dying cells are carved out of the
        # background instead of leaking into it as if they were empty medium.
        for t, masks in masks_all.items():
            np.save(os.path.join(mask_full_dir, f"mask_sk{t}.npy"),
                    masks.astype(np.uint16))
        # Cleaned masks: excluded cells zeroed out — used for all measurements.
        for t, masks in cleaned.items():
            np.save(os.path.join(mask_dir, f"mask_sk{t}.npy"),
                    masks.astype(np.uint16))

        del masks_all, cleaned
        gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("CellPose pass complete — model unloaded.\n")


# ──────────────────────────────────────────────
# Pass 3 — Analysis (CPU only)
# ──────────────────────────────────────────────

def run_analysis_pass(pos_dict, directory, save_dir, proximity_px=3,
                      no_bacteria_fields=None):
    no_bact = set(no_bacteria_fields or [])

    for (well, field), time_dict in pos_dict.items():
        if _analysis_field_done(save_dir, well, field):
            print(f"  [Analysis] {well}/{field} — already done, skipping.")
            continue

        is_control = (well, field) in no_bact
        field_dir  = os.path.join(save_dir, well, field)

        results         = []
        contact_records = []

        label = "  [NO-BACTERIA CONTROL]" if is_control else ""
        print(f"  [Analysis] {well}/{field}  ({len(time_dict)} timepoints){label}")

        for t, planes in time_dict.items():
            cell_mask = np.load(
                os.path.join(field_dir, "cell_seg", "masks", f"mask_sk{t}.npy"))

            # Full (pre-exclusion) mask defines the background: every cell
            # footprint — including excluded edge/dying cells — is carved out,
            # so removed cells contribute to neither background nor results.
            # Fall back to the cleaned mask for older runs without masks_full/.
            full_mask_path = os.path.join(
                field_dir, "cell_seg", "masks_full", f"mask_sk{t}.npy")
            if os.path.exists(full_mask_path):
                full_cell_mask = np.load(full_mask_path)
            else:
                full_cell_mask = cell_mask

            if is_control:
                bact_mask = np.zeros(cell_mask.shape, dtype=bool)
            else:
                bm = np.load(os.path.join(field_dir, "bacteria_seg", "masks", f"mask_sk{t}.npy"))
                bact_mask = bm > 0
                del bm

            ch1_raw = imread(os.path.join(directory, planes[2]['ch1'])).astype(float)
            ch2_raw = imread(os.path.join(directory, planes[2]['ch2'])).astype(float)

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
                px    = cell_mask == cell_id
                mc1   = float(ch1_corr[px].mean())
                mc2   = float(ch2_corr[px].mean())
                ratio = mc2 / mc1 if mc1 > 0 else np.nan
                results.append({
                    "time":       t,
                    "cell_id":    int(cell_id),
                    "mean_c1":    round(mc1, 4),
                    "mean_c2":    round(mc2, 4),
                    "mean_ratio": round(ratio, 4),
                })

            if not is_control:
                for cid in detect_interactions(cell_mask, bact_mask, proximity_px):
                    contact_records.append({"timepoint": t, "cell_id": cid})

            del ch1_raw, ch2_raw, ch1_corr, ch2_corr, cell_mask, full_cell_mask, bact_mask
            gc.collect()

        df = pd.DataFrame(results)
        if not df.empty:
            save_wide(df, "mean_ratio", "results_ratio.csv", field_dir)
            save_wide(df, "mean_c1",    "results_c1.csv",    field_dir)
            save_wide(df, "mean_c2",    "results_c2.csv",    field_dir)

        if contact_records:
            df_c = pd.DataFrame(contact_records)
            df_c["contact"] = 1
            cm_out = df_c.pivot_table(
                index="timepoint", columns="cell_id",
                values="contact", fill_value=0, aggfunc="max",
            )
            cm_out.columns = [f"cell_{c}" for c in cm_out.columns]
            cm_out.sort_index(inplace=True)
            cm_out.to_csv(os.path.join(field_dir, "contact_matrix.csv"))
            print(f"    [{well}/{field}] Contact matrix saved "
                  f"({df_c['cell_id'].nunique()} cells had bacterial contact)")
        else:
            print(f"    [{well}/{field}] No bacterial contacts detected.")

    print("Analysis pass complete.\n")


# ──────────────────────────────────────────────
# Pass 5 — Post-processing (overlay regeneration, no videos)
# ──────────────────────────────────────────────

def _make_deterministic_overlay(image, mask, save_path, write_id=True):
    """Grayscale image + colored, bordered, (optionally labeled) mask -> PNG."""
    img = image.astype(float)
    p2, p98 = np.percentile(img[img > 0], (2, 98)) if img.max() > 0 else (0, 1)
    img = np.clip((img - p2) / (p98 - p2 + 1e-9), 0, 1)

    colored = np.stack([img, img, img], axis=-1)
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        color = _OVERLAY_PALETTE[int(cell_id) % len(_OVERLAY_PALETTE)]
        region = mask == cell_id
        colored[region] = 0.6 * colored[region] + 0.4 * color

    boundaries = find_boundaries(mask, mode="outer")
    colored[boundaries] = [1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(colored)
    if write_id:
        for cell_id in np.unique(mask):
            if cell_id == 0:
                continue
            ys, xs = np.where(mask == cell_id)
            cx, cy = xs.mean(), ys.mean()
            ax.text(cx, cy, str(cell_id), color="yellow", fontsize=7,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    del fig, ax


def regenerate_cellpose_overlays(save_dir, directory=None, pos_dict=None):
    """(Re)create cell_seg/overlays from cell_seg/masks + raw ch2 (plane 2)
    backdrop, each overlay only once (skips existing); scoped to pos_dict wells."""
    for field_dir in sorted(glob.glob(os.path.join(save_dir, "*", "*"))):
        if not os.path.isdir(field_dir):
            continue
        well, field = os.path.basename(os.path.dirname(field_dir)), os.path.basename(field_dir)
        if pos_dict is not None and (well, field) not in pos_dict:
            continue
        mask_dir = os.path.join(field_dir, "cell_seg", "masks")
        ov_dir   = os.path.join(field_dir, "cell_seg", "overlays")
        if not os.path.isdir(mask_dir):
            continue
        os.makedirs(ov_dir, exist_ok=True)
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "mask_sk*.npy")))
        if not mask_files:
            continue
        n_made = 0
        for mf in mask_files:
            sk      = re.search(r"mask_sk(\d+)\.npy", os.path.basename(mf)).group(1)
            ov_path = os.path.join(ov_dir, f"overlay_sk{sk}.png")
            if os.path.exists(ov_path):
                continue                       # create overlays only once
            n_made += 1
            mask = np.load(mf)
            image = None
            if directory and pos_dict is not None:
                planes = pos_dict.get((well, field), {}).get(int(sk))
                if planes and 2 in planes and "ch2" in planes[2]:
                    image = imread(os.path.join(directory, planes[2]["ch2"]))
                    if image.ndim == 3:
                        image = image[..., 0]
            if image is None:
                image = np.zeros(mask.shape, dtype=np.uint16)
            _make_deterministic_overlay(image, mask, ov_path)
        if n_made:
            print(f"  [Overlays] {well}/{field}: created {n_made} overlay(s)")
        else:
            print(f"  [Overlays] {well}/{field}: all {len(mask_files)} already exist, skipping")
    print("Overlay regeneration complete.\n")


def regenerate_bacteria_overlays(save_dir, directory=None, pos_dict=None):
    """(Re)create bacteria_seg/overlays from bacteria_seg/masks + a max-projected
    ch3 backdrop, only once (skips existing); scoped to pos_dict wells."""
    for field_dir in sorted(glob.glob(os.path.join(save_dir, "*", "*"))):
        if not os.path.isdir(field_dir):
            continue
        well, field = os.path.basename(os.path.dirname(field_dir)), os.path.basename(field_dir)
        if pos_dict is not None and (well, field) not in pos_dict:
            continue
        mask_dir = os.path.join(field_dir, "bacteria_seg", "masks")
        ov_dir   = os.path.join(field_dir, "bacteria_seg", "overlays")
        if not os.path.isdir(mask_dir):
            continue
        os.makedirs(ov_dir, exist_ok=True)
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "mask_sk*.npy")))
        if not mask_files:
            continue
        n_made = 0
        for mf in mask_files:
            sk      = re.search(r"mask_sk(\d+)\.npy", os.path.basename(mf)).group(1)
            ov_path = os.path.join(ov_dir, f"overlay_sk{sk}.png")
            if os.path.exists(ov_path):
                continue
            n_made += 1
            masks = np.load(mf)
            image = None
            if directory and pos_dict is not None:
                planes = pos_dict.get((well, field), {}).get(int(sk))
                if planes:
                    try:
                        raw = max_project_channel(directory, planes, sorted(planes.keys()), "ch3")
                        image = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                    except Exception:
                        image = None
            if image is None:
                image = np.zeros(masks.shape)
            _make_deterministic_overlay(image, masks, ov_path, write_id=False)
        if n_made:
            print(f"  [Bact overlays] {well}/{field}: created {n_made}")
        else:
            print(f"  [Bact overlays] {well}/{field}: all exist, skipping")
    print("Bacteria overlay regeneration complete.\n")


def run_postprocessing_pass(save_dir, directory=None, pos_dict=None):
    # Videos are intentionally NOT generated on the cluster — only still overlays.
    regenerate_cellpose_overlays(save_dir, directory, pos_dict)
    regenerate_bacteria_overlays(save_dir, directory, pos_dict)


def _read_wide_ratio(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(int)
    df.index.name = "timepoint"
    return df


def _build_ratio_all_field(well_dir):
    """Merge every field's results_ratio.csv into one wide table (cols <field>_<cell>)."""
    parts = []
    for field_dir in sorted(glob.glob(os.path.join(well_dir, "*"))):
        if not os.path.isdir(field_dir):
            continue
        field = os.path.basename(field_dir)
        if not re.fullmatch(r"f\d+", field):
            continue
        p = os.path.join(field_dir, "results_ratio.csv")
        if not os.path.exists(p):
            continue
        d = _read_wide_ratio(p)
        d.columns = [f"{field}_{c}" for c in d.columns]
        parts.append(d)
    if not parts:
        return None
    merged = pd.concat(parts, axis=1)
    merged.sort_index(inplace=True)
    return merged


def _build_contact_time_map(well_dir):
    """{'<field>_<cell>': first_contact_timepoint} from every field's
    contact_matrix.csv in this well."""
    ctimes = {}
    for field_dir in sorted(glob.glob(os.path.join(well_dir, "*"))):
        if not os.path.isdir(field_dir):
            continue
        field = os.path.basename(field_dir)
        if not re.fullmatch(r"f\d+", field):
            continue
        p = os.path.join(field_dir, "contact_matrix.csv")
        if not os.path.exists(p):
            continue
        cm = _read_wide_ratio(p)                  # index=timepoint, cols=cell_<id>, values 0/1
        for col in cm.columns:
            hits = cm.index[cm[col] == 1]
            if len(hits):
                ctimes[f"{field}_{col}"] = int(hits.min())
    return ctimes


def _build_contact_analysis(value_df, ctime_map):
    """Contact-aligned table: rows = contacted cells, columns = time relative to
    each cell's first contact (0 = contact, negatives = before, positives =
    after), values = that cell's value at the corresponding actual timepoint.
    Only cells with a known contact time are included. Returns the DataFrame
    (or None if no contacted cell has data)."""
    rows = {}
    for col, t0 in ctime_map.items():
        if col not in value_df.columns:
            continue
        s = value_df[col]
        rel = pd.Series(s.values, index=(s.index.astype(int) - t0))  # shift so contact -> 0
        rows[col] = rel
    if not rows:
        return None
    rmin = min(int(r.index.min()) for r in rows.values())
    rmax = max(int(r.index.max()) for r in rows.values())
    cols = list(range(rmin, rmax + 1))            # contiguous, 0 sits in the middle
    out = pd.DataFrame(index=list(rows.keys()), columns=cols, dtype=float)
    for cell, rel in rows.items():
        out.loc[cell, list(rel.index)] = rel.values
    out.index.name = "cell_id"
    out.columns.name = "rel_time"
    return out


def run_delta_r_pass(save_dir, control_well, well_names):
    """ratio_all_field.csv (per well) -> control_population_mean.csv (control well)
    -> deltar_fret.csv (every well). deltaR = (ratio - control_pop)/control_pop.
    Only *well_names* are touched (shared save_dir safe across jobs)."""
    wells = {w: os.path.join(save_dir, w) for w in sorted(set(well_names))}
    if control_well not in wells:
        print(f"  [deltaR] control well {control_well!r} not in processed wells {sorted(wells)} -- skipped.")
        return
    ratio_all = {}
    for well, well_dir in wells.items():
        df = _build_ratio_all_field(well_dir)
        if df is None:
            print(f"  [deltaR] {well}: no results_ratio.csv -- skipped")
            continue
        df.to_csv(os.path.join(well_dir, "ratio_all_field.csv"))
        ratio_all[well] = df
        print(f"  [deltaR] ratio_all_field {well}: {df.shape[1]} cols x {df.shape[0]} timepoints")
    if control_well not in ratio_all:
        print(f"  [deltaR] control well {control_well!r} has no ratio data -- skipped.")
        return
    ctrl_mean = ratio_all[control_well].mean(axis=1, skipna=True)
    ctrl_mean.name = "control_population_mean"
    ctrl_mean.to_frame().to_csv(os.path.join(wells[control_well], "control_population_mean.csv"))
    print(f"  [deltaR] control_population_mean ({control_well}): mean over "
          f"{ratio_all[control_well].shape[1]} cells per timepoint")
    denom = ctrl_mean.replace(0, np.nan)
    for well, df in ratio_all.items():
        c = ctrl_mean.reindex(df.index)
        d = denom.reindex(df.index)
        deltar = df.sub(c, axis=0).div(d, axis=0)
        deltar.to_csv(os.path.join(wells[well], "deltar_fret.csv"))
        tag = "  (control -- should sit ~0)" if well == control_well else ""
        print(f"  [deltaR] deltar_fret {well}: {deltar.shape[1]} x {deltar.shape[0]}{tag}")

        # contact_analysis.csv for non-control wells (needs contact_matrix.csv
        # per field, i.e. contact detection actually ran for this well).
        if well == control_well:
            continue
        ctimes = _build_contact_time_map(wells[well])
        if not ctimes:
            print(f"  [contactAnalysis] {well}: no contact_matrix.csv found -- skipped.")
            continue
        ca = _build_contact_analysis(deltar, ctimes)
        if ca is None:
            print(f"  [contactAnalysis] {well}: no contacted cells with deltaR data -- skipped.")
            continue
        ca.to_csv(os.path.join(wells[well], "contact_analysis.csv"))
        print(f"  [contactAnalysis] {well}: {ca.shape[0]} cells x {ca.shape[1]} rel-timepoints "
              f"(rel {ca.columns.min()}..{ca.columns.max()}) -> contact_analysis.csv")
    print("delta-R FRET pass complete.\n")


def cleanup_masks(save_dir, pos_dict, no_bacteria_fields=None):
    """Remove mask directories (cell_seg/masks, cell_seg/masks_full,
    bacteria_seg/masks) for fully-done fields (results CSV + all cell overlays,
    and bacteria overlays unless the field is a no-bacteria control). Keeps
    overlays / CSVs / contact_matrix / deltaR."""
    no_bact = set(no_bacteria_fields or [])
    print("Mask cleanup (fully-done fields only):")
    for (well, field), time_dict in pos_dict.items():
        field_dir = os.path.join(save_dir, well, field)
        need   = len(time_dict)
        csv_ok = os.path.exists(os.path.join(field_dir, "results_ratio.csv"))
        n_cell = len(glob.glob(os.path.join(field_dir, "cell_seg", "overlays", "overlay_sk*.png")))
        n_bact = len(glob.glob(os.path.join(field_dir, "bacteria_seg", "overlays", "overlay_sk*.png")))
        bact_ok = (well, field) in no_bact or n_bact >= need   # controls have no bacteria
        if not (csv_ok and n_cell >= need and bact_ok):
            print(f"  [cleanup] {well}/{field}: not fully done (csv={csv_ok}, "
                  f"cell_ov={n_cell}/{need}, bact_ov={n_bact}/{need}) -- masks kept")
            continue
        deleted = 0
        for sub in (("cell_seg", "masks"), ("cell_seg", "masks_full"),
                    ("bacteria_seg", "masks")):
            d = os.path.join(field_dir, *sub)
            if os.path.isdir(d):
                deleted += len(glob.glob(os.path.join(d, "*.npy")))
                shutil.rmtree(d)            # remove the whole mask directory
        print(f"  [cleanup] {well}/{field}: removed mask dir(s) ({deleted} .npy freed)")
    print("Mask cleanup complete.\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def _filter_wells(pos_dict, wells_arg):
    """Restrict pos_dict to wells named by *wells_arg* (comma-separated). Each
    token is either a full well (r02c05) or a column (c05 -> any r<row>c05).
    'all'/empty keeps everything."""
    tokens = [t.strip() for t in (wells_arg or "").split(",") if t.strip()]
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


def process_images(directory, control_fields, exclusion, save_dir,
                   proximity_px=3, flow_threshold=0.4, cellprob_threshold=0,
                   area_change_threshold=0.5, postprocess=True,
                   control_well=None, keep_masks=False, wells=None):
    """Run the pipeline: segmentation -> analysis (+contact) -> delta-R -> overlays.

    control_well : well (e.g. 'r02c05') whose cells define the delta-R baseline
                   (None skips the delta-R pass). control_fields still marks the
                   no-bacteria control fields (contact/background).
    keep_masks   : keep mask dirs (default: delete them at the end to save space).
    """
    if exclusion not in EXCLUSION_MODES:
        raise ValueError(f"exclusion must be one of {EXCLUSION_MODES}, got {exclusion!r}")

    os.makedirs(save_dir, exist_ok=True)

    params_omnipose = {
        'channels':       None,
        'rescale':        None,
        'mask_threshold': -2,
        'flow_threshold': 0,
        'transparency':   True,
        'omni':           True,
        'cluster':        True,
        'resample':       True,
        'verbose':        False,
        'tile':           False,
        'niter':          None,
        'augment':        False,
        'affinity_seg':   True,
        'normalize':      False,
    }

    pos_dict = group_files(directory)
    if not pos_dict:
        raise SystemExit(f"No TIFF files found in {directory!r}")
    if wells:
        pos_dict = _filter_wells(pos_dict, wells)
        if not pos_dict:
            raise SystemExit(f"No wells match --wells {wells!r}")
    well_names = sorted({w for (w, f) in pos_dict})
    print(f"Found {len(pos_dict)} field(s): "
          f"{sorted(well + field for well, field in pos_dict.keys())}")
    if wells and wells != "all":
        print(f"Well filter --wells {wells!r} -> wells: {well_names}")

    # Expand control tokens ('r02c05' = whole well, 'r02c05f01' = one field)
    # against the fields actually present in this dataset.
    no_bact = resolve_control_fields(control_fields or [], pos_dict.keys())
    if no_bact:
        print(f"No-bacteria controls: {sorted(well + field for well, field in no_bact)}")
    print(f"Cell exclusion policy: {exclusion}")
    print(f"Input : {directory}")
    print(f"Output: {save_dir}")

    t_start   = time.time()
    pass_times = {}

    print(f"\n{'='*50}")
    print("Pass 1 — Bacteria segmentation (OmniPose) [max-projection + CLAHE]")
    print(f"{'='*50}")
    t0 = time.time()
    run_omnipose_pass(pos_dict, directory, save_dir, params_omnipose,
                      no_bacteria_fields=no_bact)
    pass_times["Pass 1 (OmniPose)"] = time.time() - t0
    print(f"Pass 1 took {format_duration(pass_times['Pass 1 (OmniPose)'])}.")

    print(f"\n{'='*50}")
    print("Pass 2 — Cell segmentation + tracking (CellPose)")
    print(f"{'='*50}")
    t0 = time.time()
    run_cellpose_pass(pos_dict, directory, save_dir,
                      flow_threshold=flow_threshold,
                      cellprob_threshold=cellprob_threshold,
                      area_change_threshold=area_change_threshold,
                      exclusion=exclusion)
    pass_times["Pass 2 (CellPose)"] = time.time() - t0
    print(f"Pass 2 took {format_duration(pass_times['Pass 2 (CellPose)'])}.")

    print(f"\n{'='*50}")
    print("Pass 3 — Analysis (CPU only)")
    print(f"{'='*50}")
    t0 = time.time()
    run_analysis_pass(pos_dict, directory, save_dir, proximity_px,
                      no_bacteria_fields=no_bact)
    pass_times["Pass 3 (Analysis)"] = time.time() - t0
    print(f"Pass 3 took {format_duration(pass_times['Pass 3 (Analysis)'])}.")

    print(f"\n{'='*50}")
    print("Pass 4 — delta-R FRET reports (ratio_all_field / control_population_mean / deltar_fret)")
    print(f"{'='*50}")
    t0 = time.time()
    if control_well:
        run_delta_r_pass(save_dir, control_well, well_names)
    else:
        print("  [deltaR] no --control_well given -- delta-R pass skipped.")
    pass_times["Pass 4 (deltaR)"] = time.time() - t0
    print(f"Pass 4 took {format_duration(pass_times['Pass 4 (deltaR)'])}.")

    if postprocess:
        print(f"\n{'='*50}")
        print("Pass 5 — Post-processing (overlays only, no videos)")
        print(f"{'='*50}")
        t0 = time.time()
        run_postprocessing_pass(save_dir, directory, pos_dict)
        pass_times["Pass 5 (Overlays)"] = time.time() - t0
        print(f"Pass 5 took {format_duration(pass_times['Pass 5 (Overlays)'])}.")

    if not keep_masks:
        print(f"\n{'='*50}")
        print("Cleanup — deleting mask directories to save space")
        print(f"{'='*50}")
        cleanup_masks(save_dir, pos_dict, no_bacteria_fields=no_bact)

    total = time.time() - t_start
    print(f"\n{'='*50}")
    print("Timing summary")
    print(f"{'='*50}")
    for name, secs in pass_times.items():
        print(f"  {name:<22} {format_duration(secs)}")
    print(f"  {'TOTAL':<22} {format_duration(total)}")
    print("\nAll done. Results saved to:", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four-pass FRET bacteria-cell interaction pipeline (cluster edition).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset",
                        help="Measurement folder name inside --base_dir "
                             "(or an absolute path to a folder of TIFFs).")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help=f"Root folder that contains <dataset> (default: {DEFAULT_BASE_DIR}).")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR,
                        help=f"Root folder for results (default: {DEFAULT_RESULTS_DIR}).")
    parser.add_argument("--output_name", default=None,
                        help="Result subfolder name (default: same as dataset).")
    parser.add_argument("--control_field", default="none",
                        help="No-bacteria control(s). Use a whole well (e.g. r02c05, "
                             "all its fields become control) or a single field "
                             "(e.g. r02c05f01). Comma-separate several, or 'none'.")
    parser.add_argument("--control_well", default="none",
                        help="Control WELL (e.g. r02c05) whose cells define the "
                             "delta-R baseline. well+field accepted (well used); "
                             "'none' skips the delta-R pass.")
    parser.add_argument("--exclusion", choices=EXCLUSION_MODES, default="edge",
                        help="Cell exclusion policy: none, edge, or both (default: edge).")
    parser.add_argument("--proximity_px", type=int, default=3,
                        help="Pixel dilation radius for bacteria-contact detection (default: 3).")
    parser.add_argument("--flow_threshold", type=float, default=0.4,
                        help="CellPose flow threshold (default: 0.4).")
    parser.add_argument("--cellprob_threshold", type=float, default=0,
                        help="CellPose cell probability threshold (default: 0).")
    parser.add_argument("--area_change_threshold", type=float, default=0.5,
                        help="Frame-to-frame area ratio bound for the 'both' "
                             "exclusion mode (default: 0.5).")
    parser.add_argument("--no_postprocess", action="store_true",
                        help="Skip Pass 5 (overlay regeneration).")
    parser.add_argument("--keep_masks", action="store_true",
                        help="Keep mask dirs. Default: once a field's results + overlays "
                             "exist, delete masks/masks_full/bacteria masks to save space.")
    parser.add_argument("--wells", default="all",
                        help="Restrict to these wells (comma-separated). Each token is a "
                             "full well (r02c05) or a column (c05 = every r<row>c05). "
                             "Default 'all'. Combine with --control_well so several jobs, "
                             "each with a different control, can share one save_dir.")

    args = parser.parse_args()

    # Resolve input directory: absolute path is used as-is, otherwise base_dir/dataset.
    if os.path.isabs(args.dataset):
        directory = args.dataset
        dataset_name = os.path.basename(os.path.normpath(args.dataset))
    else:
        directory = os.path.join(args.base_dir, args.dataset)
        dataset_name = args.dataset

    if not os.path.isdir(directory):
        raise SystemExit(f"Input directory does not exist: {directory}")

    # Harmony/Opera Phenix export layout: the TIFFs live in an "Images"
    # subfolder of the measurement directory, e.g.
    #   <dataset>-Measurement 1/Images/*.tiff
    # rather than directly inside <dataset>.
    for images_subdir in ("Images", "images"):
        candidate = os.path.join(directory, images_subdir)
        if os.path.isdir(candidate):
            directory = candidate
            break

    output_name = args.output_name or dataset_name
    save_dir = os.path.join(args.results_dir, output_name)

    if args.control_field.strip().lower() in ("", "none"):
        control_fields = []
    else:
        control_fields = [p.strip() for p in args.control_field.split(",") if p.strip()]

    if args.control_well.strip().lower() in ("", "none"):
        control_well = None
    else:
        m = re.match(r"(r\d+c\d+)", args.control_well.strip())
        if not m:
            raise SystemExit(f"--control_well must look like r02c05, got {args.control_well!r}")
        control_well = m.group(1)

    process_images(
        directory             = directory,
        control_fields        = control_fields,
        exclusion             = args.exclusion,
        save_dir              = save_dir,
        proximity_px          = args.proximity_px,
        flow_threshold        = args.flow_threshold,
        cellprob_threshold    = args.cellprob_threshold,
        area_change_threshold = args.area_change_threshold,
        postprocess           = not args.no_postprocess,
        control_well          = control_well,
        keep_masks            = args.keep_masks,
        wells                  = args.wells,
    )
