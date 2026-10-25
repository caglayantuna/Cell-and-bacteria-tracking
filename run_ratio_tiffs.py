"""
run_ratio_tiffs.py

Save the pixel-wise ch2/ch1 FRET ratio as TIFF images. Nothing else — no CSVs,
no mask files, no overlays.

Cluster edition, derived from run_fret_pipeline_others.py (the variant matching
this data layout: NO ch3 bacteria channel, ch1/ch2 read at plane 1 / p01).

Each timepoint is handled frame by frame, in the pipeline's own order:

  1. CellPose on ch2 @ p01 (normalised + CLAHE, diameter=100)  -> cell mask
     ... pipeline Pass 1
  2. OmniPose on ch1, cell pixels zeroed (normalised + CLAHE)  -> bacteria mask
     ... pipeline Pass 2
  3. Background = MEAN of the pixels that are neither cell nor bacterium, per
     channel, with the 5th percentile of the frame as fallback ... pipeline
     Pass 3
  4. Same order as run_fret_pipeline*_perpixel.py: subtract the background,
     median-filter ch1 and ch2 (--median_size, default 3x3, whole frame, BEFORE
     clipping), clip at 0, then ratio = ch2_f / ch1_f for pixels whose filtered
     ch1 exceeds --min_snr x the SD of the filtered ch1 over the background
     (default 3). No filtering after the division. Everything outside the cell
     mask is NaN (bacteria only shape the background, as in the pipeline).
     Written as one float32 TIFF.
  5. Cell outlines (mask boundary, 1 px, just outside each cell; shared edges
     between touching cells are marked too) are written next to it as a
     2-channel Fiji composite, see Output.

The masks are discarded once the frame's TIFF is written. Two details of the
pipeline are left out: frame-to-frame tracking (it only relabels cell IDs — the
mask footprint is identical), and the edge/dying-cell exclusion (the background
comes from the PRE-exclusion masks_full anyway). Consequence for the image:
every CellPose cell is shown, including cells touching the border.

Input layout
------------
The dataset argument may be any of:
    a folder of TIFFs                      (e.g. .../r04c06f06/)
    a folder whose subfolders hold TIFFs   (e.g. .../<measurement>/ -> r04c06f06/)
    a Harmony export with Images/          (e.g. .../<measurement>/Images/)
Every TIFF folder found is processed.

Output
------
    <parent of the TIFF folder>/ch2_ch1/ratio/ratio_<well><field>_sk###.tiff
    <parent of the TIFF folder>/ch2_ch1/boundaries/ratio_<well><field>_sk###_boundaries.tiff

i.e. a ch2_ch1 folder NEXT TO the field folder (or --out_dir), with the plain
ratio images in ratio/ and the composites with outlines in boundaries/.
ratio_*.tiff: single-channel float32; NaN outside the cells and on in-cell
pixels that fail the ch1 threshold.
*_boundaries.tiff: 2-channel float32 ImageJ composite — channel 1 = the same
ratio image, channel 2 = cell outlines (1 on the outline, 0 elsewhere). In Fiji
each channel keeps its own LUT/display range (Image > Color > Channels Tool),
so recolouring the ratio channel leaves the outlines untouched.

Resumable: a timepoint whose TIFF already exists is skipped unless --overwrite.

Usage
-----
    python run_ratio_tiffs.py <dataset> [options]

    # the representative-images field, ratio TIFFs land beside r04c06f06/
    python run_ratio_tiffs.py "/pasteur/helix/scratch/egokerku/FRET/Representative images/260722_NCI_40xW_timecourse_Y_YKK_rep1__2026-07-22T14_14_22-Measurement 1"
"""

import os
import re
import gc
import time
import argparse
from collections import defaultdict

import numpy as np
import tifffile
from scipy.ndimage import median_filter
from skimage.io import imread
from skimage.exposure import equalize_adapthist
from skimage.segmentation import find_boundaries
import torch

from cellpose_omni import models
from omnipose.gpu import use_gpu
from cellpose.models import CellposeModel


DEFAULT_BASE_DIR = "/pasteur/helix/scratch/egokerku/FRET/"

# Copied verbatim from process_images() in run_fret_pipeline_others.py
PARAMS_OMNIPOSE = {
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


# ──────────────────────────────────────────────
# Helpers — copied from run_fret_pipeline_others.py
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


def parse_filename(fname):
    """Extract well, field, focal-plane index, channel, and timepoint from a filename."""
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


def max_project_channel(directory, planes, plane_ids, channel):
    """Max-intensity projection of a raw channel across the given planes."""
    stack = np.stack([
        imread(os.path.join(directory, planes[p][channel])).astype(float)
        for p in plane_ids
    ], axis=0)
    return stack.max(axis=0)


def clahe(raw):
    """Normalise to 0-1 and CLAHE — exactly how the pipeline preprocesses before
    every segmentation call."""
    img = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    return equalize_adapthist(img, clip_limit=0.03)


# ──────────────────────────────────────────────
# Input discovery
# ──────────────────────────────────────────────

_TIFF_RE = re.compile(r'r\d+c\d+f\d+p\d+.*ch\d+sk\d+.*\.tiff?$')


def _holds_tiffs(d):
    try:
        return any(_TIFF_RE.search(f) for f in os.listdir(d))
    except OSError:
        return False


def find_tiff_dirs(root):
    """Every directory at or below *root* that holds Harmony TIFFs (sorted)."""
    if _holds_tiffs(root):
        return [root]
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames.sort()
        if _holds_tiffs(dirpath):
            found.append(dirpath)
            dirnames[:] = []          # don't descend further into a TIFF folder
    return found


# ──────────────────────────────────────────────
# Segmentation models
# ──────────────────────────────────────────────

class Segmenters:
    """CellPose (cells, ch2) + OmniPose (bacteria, ch1), loaded once and reused
    for every frame."""

    def __init__(self, args):
        self.args = args
        # omnipose's use_gpu() returns a plain bool on some builds and a
        # (device, bool) tuple on others — accept either.
        gpu = use_gpu()
        gpu = bool(gpu[1]) if isinstance(gpu, tuple) else bool(gpu)
        self.gpu = gpu and not args.cpu
        print(f"  Loading models (gpu={self.gpu}) …", flush=True)
        self.cell = CellposeModel(gpu=self.gpu)
        self.bact = models.CellposeModel(gpu=self.gpu, model_type='bact_fluor_omni')

    def cells(self, ch2_raw):
        """Pipeline Pass 1 — CellPose on ch2."""
        masks, _, _ = self.cell.eval(
            clahe(ch2_raw),
            batch_size=self.args.batch_size,
            diameter=self.args.diameter,
            flow_threshold=self.args.flow_threshold,
            cellprob_threshold=self.args.cellprob_threshold,
            normalize=False,
        )
        return masks

    def bacteria(self, ch1_proj, cell_mask):
        """Pipeline Pass 2 — OmniPose on ch1, with the cell pixels zeroed out so
        OmniPose only sees extracellular ch1 signal."""
        raw = ch1_proj.copy()
        raw[cell_mask > 0] = 0
        masks, _, _ = self.bact.eval(clahe(raw), **PARAMS_OMNIPOSE)
        return masks


# ──────────────────────────────────────────────
# One frame
# ──────────────────────────────────────────────

def ratio_frame(seg, directory, planes, plane_ids, plane, min_denom, min_snr,
                median_size):
    """Segment one timepoint, take the background from the masks, return the
    per-pixel ch2/ch1 ratio as float32 computed as in the perpixel pipeline
    (median on the background-subtracted channels, then clip, threshold, divide),
    NaN outside the cells, plus the cell-outline image."""
    ch1_raw = imread(os.path.join(directory, planes[plane]['ch1'])).astype(float)
    ch2_raw = imread(os.path.join(directory, planes[plane]['ch2'])).astype(float)

    cell_mask = seg.cells(ch2_raw)
    # ch1 is max-projected for OmniPose, as in the pipeline.
    ch1_proj  = max_project_channel(directory, planes, plane_ids, 'ch1')
    bact_mask = seg.bacteria(ch1_proj, cell_mask) > 0
    del ch1_proj
    torch.cuda.empty_cache()

    # Background = mean over pixels that are neither cell nor bacterium.
    bg_mask = (cell_mask == 0) & ~bact_mask
    if bg_mask.any():
        bg_ch1 = float(ch1_raw[bg_mask].mean())
        bg_ch2 = float(ch2_raw[bg_mask].mean())
    else:
        bg_ch1 = float(np.percentile(ch1_raw, 5))
        bg_ch2 = float(np.percentile(ch2_raw, 5))

    # Same order as run_fret_pipeline*_perpixel.py: median-filter the
    # background-subtracted channels FIRST (before clipping, so the background
    # noise estimate is not truncated at zero), then clip, then divide.
    if median_size > 1:
        ch1_f = median_filter(ch1_raw - bg_ch1, size=median_size)
        ch2_f = median_filter(ch2_raw - bg_ch2, size=median_size)
    else:
        ch1_f = ch1_raw - bg_ch1
        ch2_f = ch2_raw - bg_ch2
    if bg_mask.any():
        bg_sd_ch1 = float(ch1_f[bg_mask].std())
    else:  # no background pixels: robust SD over the whole frame
        bg_sd_ch1 = float(1.4826 * np.median(np.abs(ch1_f - np.median(ch1_f))))
    ch1_f = np.clip(ch1_f, 0, None)
    ch2_f = np.clip(ch2_f, 0, None)
    # Noise threshold on the filtered ch1 (Nordenfelt et al. 2016, Nat Commun
    # 7:13119): ratio only pixels above min_snr x the background SD.
    c1_min = max(min_snr * bg_sd_ch1, 0.0)

    valid = (ch1_f > c1_min) & (ch1_f > min_denom)
    ratio = np.full(ch1_f.shape, np.nan, dtype=np.float32)
    ratio[valid] = (ch2_f[valid] / ch1_f[valid]).astype(np.float32)

    # Only cell pixels are shown; everything else is NaN. As in the perpixel
    # pipeline, bacteria are used only for the background — bacteria pixels
    # that overlap a cell are kept, like in the per-cell mean(c2/c1).
    keep = cell_mask > 0
    ratio[~keep] = np.nan

    # 1 px outline just OUTSIDE each cell. Where two cells touch it also marks
    # the shared edge, i.e. a pixel of the neighbour. It lives in its own
    # channel, so the ratio values themselves are never altered.
    boundary = find_boundaries(cell_mask, mode="outer").astype(np.float32)

    n_cells = int(cell_mask.max())
    del ch1_raw, ch2_raw, ch1_f, ch2_f, cell_mask, bact_mask, bg_mask, valid, keep
    gc.collect()
    return ratio, boundary, n_cells


def write_with_boundaries(path, ratio, boundary):
    """2-channel ImageJ composite: channel 1 = ratio, channel 2 = cell outlines
    (0/1). Each channel has its own LUT in Fiji, so recolouring or rescaling the
    ratio channel never touches the outlines."""
    gray = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    tifffile.imwrite(path, np.stack([ratio, boundary]), imagej=True,
                     metadata={'axes': 'CYX', 'mode': 'composite',
                               'LUTs': [gray, gray]})


def process_directory(seg, directory, args):
    """Write one ratio TIFF per timepoint for every field in *directory*."""
    pos_dict = group_files(directory)
    if not pos_dict:
        print(f"  [warn] no TIFFs in {directory!r} — skipped", flush=True)
        return

    out_dir = args.out_dir or os.path.join(os.path.dirname(directory.rstrip("/")),
                                           "ch2_ch1")
    ratio_dir    = os.path.join(out_dir, "ratio")
    boundary_dir = os.path.join(out_dir, "boundaries")
    os.makedirs(ratio_dir,    exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)
    print(f"\nInput : {directory}")
    print(f"Output: {out_dir}")
    print(f"Found {len(pos_dict)} field(s): "
          f"{sorted(w + f for w, f in pos_dict.keys())}", flush=True)

    for (well, field), time_dict in pos_dict.items():
        plane_ids = sorted(next(iter(time_dict.values())).keys())
        if args.plane not in plane_ids:
            print(f"  [{well}/{field}] plane {args.plane} not present "
                  f"(available: {plane_ids}) — skipped", flush=True)
            continue
        print(f"  [{well}/{field}] plane p{args.plane:02d}, "
              f"{len(time_dict)} timepoint(s)", flush=True)

        for t, planes in time_dict.items():
            name     = f"ratio_{well}{field}_sk{t:03d}"
            out_path = os.path.join(ratio_dir, f"{name}.tiff")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"    sk{t:<3} already exists — skipped", flush=True)
                continue
            if 'ch1' not in planes[args.plane] or 'ch2' not in planes[args.plane]:
                print(f"    sk{t:<3} ch1/ch2 missing — skipped", flush=True)
                continue

            t0 = time.time()
            ratio, boundary, n_cells = ratio_frame(
                seg, directory, planes, plane_ids, args.plane, args.min_denom,
                args.min_snr, args.median_size)
            tifffile.imwrite(out_path, ratio)
            write_with_boundaries(os.path.join(boundary_dir, f"{name}_boundaries.tiff"),
                                  ratio, boundary)
            print(f"    sk{t:<3} {n_cells:3d} cells  {format_duration(time.time()-t0)}"
                  f"  -> {os.path.basename(out_path)} (+ _boundaries)", flush=True)
            del ratio, boundary
            gc.collect()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pixel-wise ch2/ch1 FRET ratio TIFFs, with the "
                    "run_fret_pipeline_others.py background (CellPose + OmniPose).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset",
                        help="Folder of TIFFs, a folder whose subfolders hold "
                             "them, or a name inside --base_dir.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help=f"Root folder containing <dataset> (default: {DEFAULT_BASE_DIR}).")
    parser.add_argument("--out_dir", default=None,
                        help="Use this folder (with ratio/ and boundaries/ inside) instead of a ch2_ch1 folder "
                             "next to each field folder.")
    parser.add_argument("--plane", type=int, default=1,
                        help="Focal plane, as in run_fret_pipeline_others.py (default: 1 = p01).")
    parser.add_argument("--min_denom", type=float, default=0.0,
                        help="ch1 pixels <= this (after background subtraction) "
                             "become NaN (default: 0).")
    parser.add_argument("--min_snr", type=float, default=3.0,
                        help="Ratio only pixels whose filtered, background-corrected "
                             "ch1 exceeds min_snr x the SD of the filtered ch1 over "
                             "the background (default: 3; 0 = off).")
    parser.add_argument("--median_size", type=int, default=3,
                        help="Median-filter window (odd) applied to ch1 and ch2 "
                             "BEFORE the division, as in the perpixel pipeline "
                             "(default: 3 = 3x3; 1 = off).")
    parser.add_argument("--diameter", type=float, default=100,
                        help="CellPose diameter (default: 100).")
    parser.add_argument("--flow_threshold", type=float, default=0.4,
                        help="CellPose flow threshold (default: 0.4).")
    parser.add_argument("--cellprob_threshold", type=float, default=0,
                        help="CellPose cell probability threshold (default: 0).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="CellPose batch size (default: 8).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute TIFFs that already exist.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if a GPU is available.")
    args = parser.parse_args()
    if args.median_size < 1 or args.median_size % 2 == 0:
        raise SystemExit(f"--median_size must be a positive odd integer, got {args.median_size}")

    directory = (args.dataset if os.path.isabs(args.dataset)
                 else os.path.join(args.base_dir, args.dataset))
    if not os.path.isdir(directory):
        raise SystemExit(f"Input directory does not exist: {directory}")

    # Harmony/Opera Phenix export layout: TIFFs may live in an Images/ subfolder.
    for images_subdir in ("Images", "images"):
        candidate = os.path.join(directory, images_subdir)
        if os.path.isdir(candidate):
            directory = candidate
            break

    tiff_dirs = find_tiff_dirs(directory)
    if not tiff_dirs:
        raise SystemExit(f"No Harmony TIFFs found under {directory!r}")
    print(f"Dataset: {directory}")
    print(f"{len(tiff_dirs)} TIFF folder(s) to process:")
    for d in tiff_dirs:
        print(f"  {d}")

    t_start = time.time()
    seg = Segmenters(args)
    for d in tiff_dirs:
        process_directory(seg, d, args)
    print(f"\nAll done in {format_duration(time.time() - t_start)}.")


if __name__ == "__main__":
    main()
