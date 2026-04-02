#!/usr/bin/env python3
"""
Unified dataset loader for YOLOv5-6D-Pose benchmark.

Usage:
    from load_dataset import load_dataset, list_datasets

    list_datasets()                              # show available datasets
    data = load_dataset("unreal_flying")         # load all sequences
    data = load_dataset("mavic2", seq_keys=["01/0101"])  # load one sequence

Each sequence is a dict with:
    t          — (N,) timestamps in seconds from 0
    pos        — (N,3) GT position
    quat       — (N,4) GT orientation xyzw
    pnp_pos    — (N,3) detection position (NaN if failed/unavailable)
    pnp_q      — (N,4) detection orientation (NaN if failed/unavailable)
    images     — list[N] of Path to image files
    metadata   — dict with dataset-specific extras (level, cam_id, etc.)
"""

import json
import re
import numpy as np
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

DATASETS = {
    "unreal_flying": {
        "type": "unreal",
        "seq_root": "data/sequences/flying",
        "json_root": "data/sequences/flying/flying",
        "folder_pattern": r"trajectory_(\d+)_(\d+)",
        "key_group": 1,  # tid is the sequence key
    },
    "unreal_step": {
        "type": "unreal",
        "seq_root": "data/sequences/acc_step",
        "json_root": "data/sequences/acc_step/acc_step",
        "folder_pattern": r"step_a(\d+)_cam(\d+)_(\d+)",
        "key_group": 3,  # last number is the sequence key
    },
    "mavic2": {
        "type": "real",
        "data_dir": "data/data/mavic2",
    },
    "phantom4": {
        "type": "real",
        "data_dir": "data/data/phantom4",
    },
}


def list_datasets():
    """Print available datasets and their paths."""
    print("Available datasets:")
    for name, cfg in DATASETS.items():
        if cfg["type"] == "unreal":
            path = REPO_ROOT / cfg["seq_root"]
        else:
            path = REPO_ROOT / cfg["data_dir"]
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name:<20} {path}")


def load_dataset(name, seq_keys=None):
    """
    Load a dataset by name.

    Args:
        name:     one of list_datasets() names
        seq_keys: optional list of sequence keys to load (None = all)

    Returns:
        dict mapping seq_key → sequence dict
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]

    if cfg["type"] == "unreal":
        return _load_unreal(cfg, seq_keys)
    elif cfg["type"] == "real":
        return _load_real(cfg, seq_keys)
    else:
        raise ValueError(f"Unknown dataset type: {cfg['type']}")


# ============================================================================
# UNREAL LOADER (flying + acc_step)
# ============================================================================

def _load_unreal(cfg, seq_keys=None):
    seq_root = REPO_ROOT / cfg["seq_root"]
    json_root = REPO_ROOT / cfg["json_root"]

    if not json_root.exists():
        raise FileNotFoundError(f"JSON root not found: {json_root}")

    # Load all sequence JSONs
    all_jsons = {}
    for d in sorted(json_root.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
        seq_json = d / "sequence.json"
        if not seq_json.exists():
            continue
        with open(seq_json) as f:
            jdata = json.load(f)
        key = list(jdata.keys())[0]
        all_jsons[key] = jdata[key]

    # Build folder mapping: seq_key → folder path
    folder_pattern = re.compile(cfg["folder_pattern"])
    key_group = cfg["key_group"]
    key_to_folder = {}

    for d in seq_root.iterdir():
        if not d.is_dir():
            continue
        m = folder_pattern.match(d.name)
        if not m:
            continue
        images_dir = d / "drone" / "images"
        if not images_dir.exists():
            continue
        seq_key = m.group(key_group)
        key_to_folder[seq_key] = d

    # Filter keys
    if seq_keys is not None:
        keys = [str(k) for k in seq_keys]
    else:
        keys = sorted(all_jsons.keys(), key=lambda x: int(x) if x.isdigit() else x)

    # Build output
    data = {}
    for key in keys:
        if key not in all_jsons:
            continue

        entry = all_jsons[key]

        # Find images
        images = []
        if key in key_to_folder:
            images_dir = key_to_folder[key] / "drone" / "images"
            images = sorted(
                list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
                key=lambda f: int(f.stem)
            )

        # Timestamps
        t = np.array(entry["t"])
        if len(t) > 0:
            t = t - t[0]  # seconds from 0

        N = len(t)

        # PnP fields — may be absent or full of NaN
        if "pnp_pos" in entry and len(entry["pnp_pos"]) == N:
            pnp_pos = np.array(entry["pnp_pos"])
        else:
            pnp_pos = np.full((N, 3), np.nan)

        if "pnp_q" in entry and len(entry["pnp_q"]) == N:
            pnp_q = np.array(entry["pnp_q"])
        else:
            pnp_q = np.full((N, 4), np.nan)

        # Build metadata (everything not in the standard fields)
        standard_keys = {"t", "pos", "quat", "pnp_pos", "pnp_q",
                         "vel", "rts_vel", "rts_acc", "omega_b",
                         "cam_pos", "cam_q"}
        metadata = {k: v for k, v in entry.items() if k not in standard_keys}

        data[key] = {
            "t": t,
            "pos": np.array(entry["pos"]),
            "quat": np.array(entry["quat"]),
            "pnp_pos": pnp_pos,
            "pnp_q": pnp_q,
            "images": images,
            "metadata": metadata,
        }

        # Add optional GT fields if present
        for field in ["rts_vel", "rts_acc", "omega_b", "cam_pos", "cam_q"]:
            if field in entry:
                data[key][field] = np.array(entry[field])

    return data


# ============================================================================
# REAL-WORLD LOADER (mavic2 + phantom4)
# ============================================================================

def _load_real(cfg, seq_keys=None):
    data_dir = REPO_ROOT / cfg["data_dir"]
    img_root = data_dir / "JPEGImages"
    label_root = data_dir / "labels"

    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")

    # Discover all session/sequence pairs
    trials = []
    for session_dir in sorted(img_root.iterdir()):
        if not session_dir.is_dir():
            continue
        for seq_dir in sorted(session_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            trial_key = f"{session_dir.name}/{seq_dir.name}"
            trials.append((session_dir.name, seq_dir.name, trial_key))

    # Filter
    if seq_keys is not None:
        seq_keys_set = set(str(k) for k in seq_keys)
        trials = [(s, q, k) for s, q, k in trials if k in seq_keys_set]

    data = {}
    for session, sequence, trial_key in trials:
        img_dir = img_root / session / sequence
        lbl_dir = label_root / session / sequence

        if not lbl_dir.exists():
            continue

        image_files = sorted(img_dir.glob("*.jpg"), key=lambda f: int(f.stem))
        if not image_files:
            continue

        # Load per-frame GT labels
        timestamps_ns = []
        gt_pos_list = []
        gt_q_list = []
        cam_pos_list = []
        cam_q_list = []
        valid_images = []

        for img_path in image_files:
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            with open(label_path) as f:
                vals = f.readline().strip().split()

            if len(vals) < 16:
                continue

            ts_cam = int(vals[0])
            cam_t = np.array([float(v) for v in vals[1:4]])
            cam_q = np.array([float(v) for v in vals[4:8]])
            uav_t = np.array([float(v) for v in vals[9:12]])
            uav_q = np.array([float(v) for v in vals[12:16]])

            timestamps_ns.append(ts_cam)
            cam_pos_list.append(cam_t)
            cam_q_list.append(cam_q)
            gt_pos_list.append(uav_t)
            gt_q_list.append(uav_q)
            valid_images.append(img_path)

        if not timestamps_ns:
            continue

        ts = np.array(timestamps_ns, dtype=np.int64)
        t = (ts - ts[0]) / 1e9

        N = len(t)
        data[trial_key] = {
            "t": t,
            "pos": np.array(gt_pos_list),
            "quat": np.array(gt_q_list),
            "pnp_pos": np.full((N, 3), np.nan),
            "pnp_q": np.full((N, 4), np.nan),
            "images": valid_images,
            "cam_pos": np.array(cam_pos_list),
            "cam_q": np.array(cam_q_list),
            "metadata": {"session": session, "sequence": sequence},
        }

    return data


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        list_datasets()
        print("\nUsage: python load_dataset.py <dataset_name> [seq_key]")
        sys.exit(0)

    name = sys.argv[1]
    seq_keys = [sys.argv[2]] if len(sys.argv) > 2 else None

    data = load_dataset(name, seq_keys=seq_keys)
    print(f"\nLoaded {len(data)} sequences from '{name}'")

    for key in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x)[:5]:
        seq = data[key]
        n = len(seq["t"])
        n_imgs = len(seq["images"])
        has_pnp = not np.all(np.isnan(seq["pnp_pos"]))
        print(f"  {key}: {n} frames, {n_imgs} images, pnp={'yes' if has_pnp else 'no'}")

    if len(data) > 5:
        print(f"  ... and {len(data) - 5} more")
