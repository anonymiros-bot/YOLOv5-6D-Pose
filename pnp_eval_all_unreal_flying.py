#!/usr/bin/env python3
# YOLO6D → PnP → WORLD-FRAME POSE FOR SIMULATED flying SEQUENCES
#
# PURPOSE:
#   Run YOLOv5-6D on all 140 flying trajectories, solve PnP per frame,
#   transform to world frame using known camera poses, save results JSON.
#   Original sequence JSONs are never modified.
#
# PIPELINE (per frame):
#   image → letterbox → YOLOv5-6D → 9 keypoints (pixels)
#     → solvePnP + refineLM on 8 corners → T_cam_mav (4×4)
#       → T_world_cam (from JSON cam_pos/cam_q) @ T_cam_mav → world pose
#         → pnp_pos (3,), pnp_q xyzw (4,)  [NaN if detection/PnP failed]
#
# CONSTANTS:
#   K:            fx=fy=960, cx=960, cy=540 (simulated pinhole)
#   DIST_COEFFS:  zeros (ideal, no distortion)
#   CORNERS_BODY: 9×3 from mesh 124.ply (centroid + 8 bbox corners)
#   WEIGHTS:      runs/train/exp9/weights/best.pt
#
# INPUT:
#   flying/flying/{k}/sequence.json — 140 individual GT sequence files
#     each wraps {k: {t, pos, quat, vel, cam_pos, cam_q, pnp_pos(NaN), ...}}
#   flying/trajectory_{k}_{suffix}/drone/images/*.jpg — rendered frames
#   Linking: trajectory folder tid parsed from name → matches JSON key k
#
# OUTPUT:
#   estimation_data/unreal/flying/pnp_results.json
#   Flat dict keyed "0"–"139", same fields as original + pnp_pos/pnp_q filled.
#   Console: per-sequence success counts + summary table.

import copy
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_REPO = str(Path.home() / "git/YOLOv5-6D-Pose")
WEIGHTS = str(Path.home() / "Desktop/YOLOv5-6D-Pose/runs/train/unreal/weights/best.pt")

SEQ_ROOT  = Path.home() / "Desktop/YOLOv5-6D-Pose/data/sequences/flying"
JSON_ROOT = SEQ_ROOT / "flying"   # flying/flying/{k}/sequence.json
OUT_DIR   = Path.home() / "Desktop/rl_tracking/examples/deep_eagle/estimation_data/unreal/flying"

DEVICE = ""  # "" = auto (GPU if available), "cpu" for CPU

# ============================================================================
# CAMERA + 3D MODEL (simulated pinhole)
# ============================================================================

K = np.array([[960, 0, 960], [0, 960, 540], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros(5, dtype=np.float64)

# Mesh extents from 124.ply
MIN_X, MAX_X = -0.23410146, 0.23505676
MIN_Y, MAX_Y = -0.23363571, 0.23412720
MIN_Z, MAX_Z = -0.15853004, 0.22499983

CORNERS_BODY = np.array([
    [(MIN_X+MAX_X)/2, (MIN_Y+MAX_Y)/2, (MIN_Z+MAX_Z)/2],  # centroid
    [MIN_X, MIN_Y, MIN_Z], [MIN_X, MIN_Y, MAX_Z],
    [MIN_X, MAX_Y, MIN_Z], [MIN_X, MAX_Y, MAX_Z],
    [MAX_X, MIN_Y, MIN_Z], [MAX_X, MIN_Y, MAX_Z],
    [MAX_X, MAX_Y, MIN_Z], [MAX_X, MAX_Y, MAX_Z],
], dtype=np.float64)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(weights, device_str):
    import torch
    sys.path.insert(0, YOLO_REPO)
    from models.experimental import attempt_load
    from utils.torch_utils import select_device

    device = select_device(device_str)
    model = attempt_load(weights, map_location=device)
    model.to(device)
    model.eval()
    print(f"  Model device: {next(model.parameters()).device}")
    return model, device


def predict_corners(model, image_path, device, img_size=640, conf_thresh=0.01):
    import torch
    sys.path.insert(0, YOLO_REPO)
    from utils.datasets import letterbox
    from utils.general import check_img_size, scale_coords
    from utils.pose_utils import box_filter

    img0 = cv2.imread(str(image_path))
    if img0 is None:
        return None, 0.0
    h0, w0 = img0.shape[:2]

    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    img, ratio, pad = letterbox(img0, img_size, stride=stride, auto=False)
    shape = (h0, w0)
    shapes = ((h0, w0), (ratio, pad))

    img_t = img[:, :, ::-1].transpose(2, 0, 1)
    img_t = np.ascontiguousarray(img_t)
    img_t = torch.from_numpy(img_t).to(device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred, _ = model(img_t)
    pred = box_filter(pred, conf_thres=conf_thresh, max_det=10)

    if pred is None or len(pred) == 0 or len(pred[0]) == 0:
        return None, 0.0

    det = pred[0][0].clone().cpu()
    confidence = float(det[18])
    corners_pred = det[:18].reshape(1, 18)
    scale_coords(img_t.shape[2:], corners_pred, shape, shapes[1])
    return corners_pred[0].numpy().reshape(9, 2), confidence


# ============================================================================
# PnP + WORLD FRAME TRANSFORM
# ============================================================================

def solve_pnp(corners_2d):
    """PnP on 8 corners (skip centroid). Returns T_cam_mav (4x4) or None."""
    obj_pts = CORNERS_BODY[1:].astype(np.float64)
    img_pts = corners_2d[1:].astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    # Refine with LM
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, DIST_COEFFS, rvec, tvec)

    R_cam_mav, _ = cv2.Rodrigues(rvec)
    T_cam_mav = np.eye(4)
    T_cam_mav[:3, :3] = R_cam_mav
    T_cam_mav[:3, 3] = tvec.flatten()
    return T_cam_mav


def pnp_to_world(T_cam_mav, cam_pos, cam_q_xyzw):
    """
    Transform PnP result from camera frame to world frame.

    cam_pos:     (3,) camera position in world
    cam_q_xyzw:  (4,) camera orientation in world (scipy xyzw)
    Camera convention: z forward, x right, y down (OpenCV)

    Returns: (pos_world (3,), q_world_xyzw (4,))
    """
    R_world_cam = Rot.from_quat(cam_q_xyzw).as_matrix()
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_world_cam
    T_world_cam[:3, 3] = cam_pos

    T_world_mav = T_world_cam @ T_cam_mav

    pos_world = T_world_mav[:3, 3]
    q_world = Rot.from_matrix(T_world_mav[:3, :3]).as_quat()  # xyzw
    return pos_world, q_world


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ── 1) Load all sequence JSONs from flying/flying/{k}/sequence.json ──
    dataset = {}
    for d in sorted(JSON_ROOT.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else -1):
        seq_json = d / "sequence.json"
        if not seq_json.exists():
            continue
        with open(seq_json) as f:
            jdata = json.load(f)
        json_key = list(jdata.keys())[0]
        assert json_key == d.name, \
            f"Folder/key mismatch: folder={d.name} but JSON key={json_key}"
        assert json_key not in dataset, \
            f"Duplicate JSON key: {json_key}"
        dataset[json_key] = jdata[json_key]
    print(f"Loaded {len(dataset)} sequences from {JSON_ROOT}")

    # ── 2) Build tid → image folder mapping (1:1, assert no duplicates) ──
    pat = re.compile(r"trajectory_(\d+)_(\d+)")
    tid_to_folder = {}
    for d in SEQ_ROOT.iterdir():
        m = pat.match(d.name)
        if m and (d / "drone" / "images").exists():
            tid = m.group(1)
            assert tid not in tid_to_folder, \
                f"Duplicate image folder for tid={tid}: {tid_to_folder[tid]} vs {d.name}"
            tid_to_folder[tid] = d.name
    print(f"Found {len(tid_to_folder)} image folders")

    # Verify coverage
    missing = sorted(set(dataset.keys()) - set(tid_to_folder.keys()), key=int)
    orphans = sorted(set(tid_to_folder.keys()) - set(dataset.keys()), key=int)
    if missing:
        print(f"WARNING: {len(missing)} JSON keys have no image folder: {missing[:10]}...")
    if orphans:
        print(f"WARNING: {len(orphans)} image folders have no JSON: {orphans[:10]}...")

    # Create output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 3) Load model ──
    print("Loading model...")
    model, device = load_model(WEIGHTS, DEVICE)
    print()

    # ── 4) Process all sequences ──
    output = {}

    for seq_key in tqdm(sorted(dataset.keys(), key=int), desc="Sequences"):
        # ── Look up data and folder for this sequence ──
        if seq_key not in tid_to_folder:
            tqdm.write(f"  {seq_key}: no image folder found, skipping")
            continue

        folder_name = tid_to_folder[seq_key]
        entry = dataset[seq_key]
        images_dir = SEQ_ROOT / folder_name / "drone" / "images"

        image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
            key=lambda f: int(f.stem)
        )

        N = len(entry["t"])
        cam_pos_all = np.array(entry["cam_pos"])   # (N, 3)
        cam_q_all = np.array(entry["cam_q"])        # (N, 4) xyzw

        # ── Safety checks ──
        if np.any(np.isnan(cam_pos_all)):
            tqdm.write(f"  {seq_key} ({folder_name}): cam_pos is NaN, skipping")
            continue

        if len(image_files) != N:
            tqdm.write(f"  {seq_key} ({folder_name}): imgs={len(image_files)} != N={N}, skipping")
            continue

        assert cam_pos_all.shape == (N, 3), f"{seq_key}: cam_pos shape {cam_pos_all.shape}"
        assert cam_q_all.shape == (N, 4),   f"{seq_key}: cam_q shape {cam_q_all.shape}"

        # ── Run YOLO6D + PnP per frame ──
        pnp_pos = np.full((N, 3), np.nan)
        pnp_q = np.full((N, 4), np.nan)
        n_ok = 0

        for i, img_path in enumerate(image_files):
            corners_2d, confidence = predict_corners(model, img_path, device)
            if corners_2d is None:
                continue

            T_cam_mav = solve_pnp(corners_2d)
            if T_cam_mav is None:
                continue

            pos_w, q_w = pnp_to_world(T_cam_mav, cam_pos_all[i], cam_q_all[i])
            pnp_pos[i] = pos_w
            pnp_q[i] = q_w
            n_ok += 1

        # ── Build output entry: deep copy original, overwrite PnP fields ──
        out_entry = copy.deepcopy(entry)
        out_entry["pnp_pos"] = pnp_pos.tolist()
        out_entry["pnp_q"] = pnp_q.tolist()
        out_entry["image_dir"] = folder_name

        output[seq_key] = out_entry
        tqdm.write(f"  {seq_key} ({folder_name}): {n_ok}/{N} PnP OK")

    # ── 5) Save output JSON ──
    out_path = OUT_DIR / "pnp_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved: {out_path}")
    print(f"  {len(output)} sequences, JSON size: {out_path.stat().st_size / 1e6:.1f} MB")

    # ── 6) Print summary ──
    print(f"\n{'='*75}")
    print(f"{'key':>4}  {'folder':<30}  {'level':>5}  {'arena':>6}  {'ok/N':>7}")
    print(f"{'-'*75}")
    total_ok = 0
    total_n = 0
    for seq_key in sorted(output.keys(), key=int):
        e = output[seq_key]
        N = len(e["t"])
        n_ok = sum(1 for p in e["pnp_pos"] if not any(np.isnan(p)))
        total_ok += n_ok
        total_n += N
        print(f"{seq_key:>4}  {e.get('image_dir','?'):<30}  {e['level']:>5}  ±{e['arena']:>4}  {n_ok:>3}/{N}")
    print(f"{'-'*75}")
    print(f"{'TOTAL':>4}  {'':30}  {'':>5}  {'':>6}  {total_ok:>3}/{total_n}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()