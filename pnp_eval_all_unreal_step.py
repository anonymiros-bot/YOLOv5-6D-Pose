# YOLO6D → PnP → WORLD-FRAME POSE FOR SIMULATED acc_step SEQUENCES
#
# PURPOSE:
#   Run YOLOv5-6D on all 35 acc_step sequences, solve PnP per frame,
#   transform to world frame using known camera poses, save results JSON.
#   Original dataset JSON is never modified.
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
#   dataset_acc_step.json — 35 sequences with GT poses + cam_pos/cam_q
#   acc_step/{folder}/drone/images/*.jpg — 150 rendered frames per sequence
#
# OUTPUT:
#   estimation_data/unreal/acc_step/pnp_results.json
#   Same structure as original, pnp_pos/pnp_q replaced with estimates.
#   Console: per-sequence success counts + summary table.

#!/usr/bin/env python3
"""
PnP evaluation on simulated acc_step sequences.

Reads:  dataset_acc_step.json (original, not modified)
        acc_step/{folder}/drone/images/*.jpg  (rendered frames)
        acc_step/{folder}/drone/labels/*.txt  (YOLO6D labels)

Writes: estimation_data/unreal/acc_step/pnp_results.json
        Same structure as original, but pnp_pos / pnp_q filled from PnP.
"""

import numpy as np
import cv2
import sys
import json
import re
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_REPO = str(Path.home() / "git/YOLOv5-6D-Pose")
WEIGHTS = str(Path.home() / "Desktop/YOLOv5-6D-Pose/runs/train/exp9/weights/best.pt")

SEQ_ROOT = Path.home() / "Desktop/YOLOv5-6D-Pose/data/sequences/acc_step"
JSON_PATH = Path.home() / "Desktop/rl_tracking/examples/deep_eagle/dataset_acc_step.json"
OUT_DIR = Path.home() / "Desktop/rl_tracking/examples/deep_eagle/estimation_data/unreal/acc_step"

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
    # Load original JSON (read-only)
    with open(JSON_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} sequences from {JSON_PATH}")

    # Create output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model, device = load_model(WEIGHTS, DEVICE)
    print()

    # Build output dict — same keys as original
    output = {}

    for seq_key in tqdm(sorted(dataset.keys(), key=int), desc="Sequences"):
        entry = dataset[seq_key]

        # Get image folder
        image_dir_name = entry.get("image_dir")
        if image_dir_name is None:
            # Fallback: match by level + cam_id
            level, cam_id = entry["level"], entry["cam_id"]
            pattern = re.compile(rf"step_a{level}_cam{cam_id}_\d+")
            candidates = [d for d in SEQ_ROOT.iterdir() if d.is_dir() and pattern.match(d.name)]
            if not candidates:
                tqdm.write(f"  {seq_key}: no image folder found for level={level} cam={cam_id}")
                continue
            image_dir_name = candidates[0].name

        images_dir = SEQ_ROOT / image_dir_name / "drone" / "images"
        labels_dir = SEQ_ROOT / image_dir_name / "drone" / "labels"

        # Get sorted image files — should correspond 1:1 to JSON time array
        image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
            key=lambda f: int(f.stem)
        )

        N = len(entry["t"])
        cam_pos_all = np.array(entry["cam_pos"])   # (N, 3)
        cam_q_all = np.array(entry["cam_q"])        # (N, 4) xyzw

        if len(image_files) != N:
            tqdm.write(f"  {seq_key} ({image_dir_name}): image count {len(image_files)} != JSON timesteps {N}, skipping")
            continue

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

        # Build output entry — copy original structure, replace pnp fields
        out_entry = {}
        for k, v in entry.items():
            if k == "pnp_pos":
                out_entry["pnp_pos"] = pnp_pos.tolist()
            elif k == "pnp_q":
                out_entry["pnp_q"] = pnp_q.tolist()
            else:
                out_entry[k] = v

        # Also add if keys didn't exist
        if "pnp_pos" not in out_entry:
            out_entry["pnp_pos"] = pnp_pos.tolist()
        if "pnp_q" not in out_entry:
            out_entry["pnp_q"] = pnp_q.tolist()

        output[seq_key] = out_entry

        tqdm.write(f"  {seq_key} ({image_dir_name}): {n_ok}/{N} PnP OK")

    # Save output JSON
    out_path = OUT_DIR / "pnp_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved: {out_path}")
    print(f"  {len(output)} sequences, JSON size: {out_path.stat().st_size / 1e6:.1f} MB")

    # Print summary
    print(f"\n{'='*70}")
    print(f"{'key':>4}  {'folder':<30}  {'level':>5}  {'cam':>3}  {'ok/N':>7}")
    print(f"{'-'*70}")
    total_ok = 0
    total_n = 0
    for seq_key in sorted(output.keys(), key=int):
        e = output[seq_key]
        N = len(e["t"])
        n_ok = sum(1 for p in e["pnp_pos"] if not any(np.isnan(p)))
        total_ok += n_ok
        total_n += N
        print(f"{seq_key:>4}  {e.get('image_dir','?'):<30}  {e['level']:>5}  {e['cam_id']:>3}  {n_ok:>3}/{N}")
    print(f"{'-'*70}")
    print(f"{'TOTAL':>4}  {'':30}  {'':>5}  {'':>3}  {total_ok:>3}/{total_n}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()