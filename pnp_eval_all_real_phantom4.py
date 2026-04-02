#!/usr/bin/env python3

# FULL-DATASET PnP EVALUATION: ALL TRIALS → STRUCTURED RESULTS + PLOTS (V3)
#
# PURPOSE:
#   Run the complete YOLOv5-6D → PnP → VICON world-frame pipeline on EVERY
#   trial in the dataset, collect per-frame pose estimates into a structured
#   dict, save as pickle, and produce per-trial time-series plots plus an
#   overall summary table and JSON.
#
# PIPELINE (per frame):
#   image → YOLOv5-6D → 9 predicted keypoints (pixels)
#     → solvePnP (ITERATIVE) + refineLM → T_cam_mav
#       → inv(VICON2CAM_T) @ T_cam_mav → position + quaternion in VICON frame
#         → compare vs GT uav_t / uav_q → position error (m), angular error (°)
#   Failed detections or PnP failures → NaN-filled rows (frame still recorded)
#
# OUTPUT DATA STRUCTURE:
#   data[session][sequence] = {
#       "time_s":        (N,)     — seconds from first frame (derived from cam timestamp)
#       "gt_pos":        (N, 3)   — GT UAV position in VICON frame
#       "gt_q":          (N, 4)   — GT UAV quaternion xyzw in VICON frame
#       "gt_cam_pos":    (N, 3)   — GT camera position in VICON frame
#       "gt_cam_q":      (N, 4)   — GT camera quaternion xyzw in VICON frame
#       "pnp_pos":       (N, 3)   — PnP-estimated position in VICON frame (NaN if failed)
#       "pnp_q":         (N, 4)   — PnP-estimated quaternion xyzw (NaN if failed)
#       "pred_corners":  (N,9,2)  — predicted 2D corners in original image pixels
#       "gt_corners":    (N,9,2)  — GT 2D corners from labels_yolo6d (pixels, denormalized)
#       "pnp_reproj":    (N,9,2)  — PnP reprojected corners (pixels)
#       "confidence":    (N,)     — detection confidence (0 if no detection)
#       "reproj_err_px": (N,)     — mean reprojection error over 9 points
#       "corner_err_px": (N,)     — mean L2 pred vs GT corners (8 corners, no centroid)
#       "pos_err_m":     (N,)     — position error in meters
#       "ang_err_deg":   (N,)     — angular error in degrees
#       "frame_names":   list[N]  — original frame filenames (stems)
#       "pnp_success":   (N,) bool — whether PnP succeeded for this frame
#   }
#
# INPUT DIRECTORY LAYOUT (EXPECTED):
#   DATA_DIR/
#     JPEGImages/<session>/<sequence>/*.jpg
#     labels/<session>/<sequence>/*.txt         — VICON GT (16 values: 2× timestamp+t+q)
#     labels_yolo6d/<session>/<sequence>/*.txt  — YOLO6D normalized keypoint labels
#   Trials auto-discovered: every (session, sequence) pair under JPEGImages/
#
#   GT label format:
#     ts_cam(ns) tx ty tz qx qy qz qw  ts_uav(ns) tx ty tz qx qy qz qw
#   YOLO label format:
#     class x0 y0 x1 y1 ... x8 y8 [bbox]   (normalized to image dims)
#
# CONSTANTS:
#   K:             (3,3) — Phantom 4 intrinsics (fx≈fy≈1979, cx≈977, cy≈534)
#   DIST_COEFFS:   (5,)  — [k1, k2, p1, p2, k3]
#   VICON2CAM_T:   (4,4) — rigid VICON→camera transform from calibration
#   CORNERS_BODY:  (9,3) — Phantom 4 3D bbox in MAV body frame (meters)
#
# DOES:
#   discover_trials():    scan JPEGImages/ for all session/sequence pairs
#   process_trial():      run full pipeline on one trial, return structured dict
#                         NaN-fills frames with no detection or PnP failure
#   load_yolo_label():    denormalize directly by image dims (1920×1080), no letterbox
#   solve_pnp_camera_frame(): PnP on 8 corners + LM refinement, reproj on all 9
#   pnp_to_world_frame(): inv(VICON2CAM_T) @ T_cam_mav → world position + quaternion
#   make_quat_continuous(): flip quaternion signs to avoid ±q plotting discontinuities
#   plot_trial():         3 figures per trial:
#                           position (X/Y/Z vs time, GT black + PnP red)
#                           quaternion (qx/qy/qz/qw vs time)
#                           errors (position mm + angular ° vs time)
#
#   main():
#     - Discovers all trials, loads model once
#     - Processes each trial, builds nested data dict
#     - Saves results.pkl (full numpy data) + summary.json (aggregate stats)
#     - Saves per-trial plots to plots/ subdirectory
#     - Prints summary table: session/seq, OK/total frames, mean pos mm, mean ang °
#
# OUTPUT:
#   {model_name}_{weights_name}/
#     results.pkl    — pickle of data[session][sequence] with all numpy arrays
#     summary.json   — overall + per-trial aggregate metrics
#     plots/
#       {session}_{sequence}_position.png
#       {session}_{sequence}_quaternion.png
#       {session}_{sequence}_errors.png
#
# DEPENDENCIES:
#   numpy, opencv, scipy, matplotlib, torch, tqdm, json, pickle
#   YOLOv5-6D-Pose repo
#
# NOTES:
#   - Timestamps are nanoseconds in labels; converted to seconds from trial start for plots.
#   - YOLO GT corners denormalized by fixed 1920×1080 (no letterbox undo needed here
#     since these are used only for pixel-space comparison, not for PnP).
#   - Quaternion convention: xyzw throughout (scipy default).
#   - N_IMAGES=None processes all frames; set integer to cap per trial.

"""
PnP Evaluation on Phantom 4 / Mavic 2 Dataset - V3

Produces structured dict:
    data[session][sequence] = {
        "time_s":           [N],        # seconds from 0
        "gt_pos":           [N, 3],     # UAV position in VICON frame
        "gt_q":             [N, 4],     # UAV quaternion xyzw in VICON frame
        "gt_cam_pos":       [N, 3],     # Camera position in VICON frame
        "gt_cam_q":         [N, 4],     # Camera quaternion xyzw in VICON frame
        "pnp_pos":          [N, 3],     # PnP estimated position in VICON frame
        "pnp_q":            [N, 4],     # PnP estimated quaternion xyzw in VICON frame
        "pred_corners":     [N, 9, 2],  # Predicted 2D corners (pixels)
        "gt_corners":       [N, 9, 2],  # GT 2D corners from labels_yolo6d (pixels)
        "pnp_reproj":       [N, 9, 2],  # PnP reprojected corners (pixels)
        "confidence":       [N],        # Detection confidence
        "reproj_err_px":    [N],        # Mean reprojection error
        "corner_err_px":    [N],        # Mean corner pixel error (pred vs GT)
        "pos_err_m":        [N],        # Position error in meters
        "ang_err_deg":      [N],        # Angular error in degrees
        "frame_names":      [N],        # Original frame names
        "pnp_success":      [N],        # Whether PnP succeeded
    }

Saves to: {model_name}_{weights_name}/
Plots to: {model_name}_{weights_name}/plots/
"""

import numpy as np
import cv2
import sys
import json
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent
YOLO_REPO = str(REPO_ROOT)
WEIGHTS   = str(REPO_ROOT / "runs/train/phantom4/weights/best.pt")
DATA_DIR  = REPO_ROOT / "data/data/phantom4"
OUT_DIR   = REPO_ROOT / "estimation_data/real/phantom4"

MODEL_NAME = "phantom4"
N_IMAGES   = None
DEVICE     = ""

# ============================================================================
# CONSTANTS
# ============================================================================

FX, FY, CX, CY = 1979.4, 1979.1, 976.8189, 533.9717
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

DIST_COEFFS = np.array([-0.2306, 0.1497, -0.00089582, -0.00086321, -0.0522086113480487],
                        dtype=np.float64)

VICON2CAM_T = np.array([
    [0.6685859,  -0.74342,     0.01787715, -0.3814142759180792],
    [0.01558769, -0.01002444, -0.99982825,  1.604836888783723],
    [0.74347153,  0.66874974,  0.004886,    3.035574156448842],
    [0,           0,           0,            1]
], dtype=np.float64)

# Phantom 4 3D bounding box corners in MAV body frame
MIN_X, MAX_X = -0.18, 0.16
MIN_Y, MAX_Y = -0.16, 0.18
MIN_Z, MAX_Z = -0.17, 0.06

CORNERS_BODY = np.array([
    [(MIN_X + MAX_X) / 2, (MIN_Y + MAX_Y) / 2, (MIN_Z + MAX_Z) / 2],
    [MIN_X, MIN_Y, MIN_Z], [MIN_X, MIN_Y, MAX_Z],
    [MIN_X, MAX_Y, MIN_Z], [MIN_X, MAX_Y, MAX_Z],
    [MAX_X, MIN_Y, MIN_Z], [MAX_X, MIN_Y, MAX_Z],
    [MAX_X, MAX_Y, MIN_Z], [MAX_X, MAX_Y, MAX_Z],
], dtype=np.float64)


# ============================================================================
# LABEL LOADING
# ============================================================================

def load_gt_label(label_path):
    """
    Load GT from labels/.
    Format: ts_cam tx ty tz rx ry rz rw ts_uav tx ty tz rx ry rz rw
    Returns dict with timestamps (nanoseconds), camera and UAV poses.
    """
    with open(label_path) as f:
        vals = f.readline().strip().split()

    ts_cam = int(vals[0])
    cam_t = np.array([float(v) for v in vals[1:4]])
    cam_q = np.array([float(v) for v in vals[4:8]])

    ts_uav = int(vals[8])
    uav_t = np.array([float(v) for v in vals[9:12]])
    uav_q = np.array([float(v) for v in vals[12:16]])

    return {
        "ts_cam_ns": ts_cam,
        "ts_uav_ns": ts_uav,
        "cam_t": cam_t,
        "cam_q_xyzw": cam_q,
        "uav_t": uav_t,
        "uav_q_xyzw": uav_q,
    }


def load_yolo_label(label_path, img_w=1920, img_h=1080):
    """Load GT 2D corners from labels_yolo6d/. Returns 9x2 pixel coords."""
    with open(label_path) as f:
        vals = [float(x) for x in f.readline().strip().split()]
    corners_norm = np.array(vals[1:19]).reshape(9, 2)
    corners_px = corners_norm.copy()
    corners_px[:, 0] *= img_w
    corners_px[:, 1] *= img_h
    return corners_px


# ============================================================================
# MODEL
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
        return None, None, "Failed to load image"
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
        return None, None, "No detections"

    det = pred[0][0].clone().cpu()
    confidence = float(det[18])
    corners_pred = det[:18].reshape(1, 18)
    scale_coords(img_t.shape[2:], corners_pred, shape, shapes[1])
    return corners_pred[0].numpy().reshape(9, 2), confidence, None


# ============================================================================
# PnP
# ============================================================================

def solve_pnp_camera_frame(corners_2d, use_distortion=True):
    obj_pts = CORNERS_BODY[1:].astype(np.float64)
    img_pts = corners_2d[1:].astype(np.float64)
    dist = DIST_COEFFS if use_distortion else np.zeros(5)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)
    R_cam_mav, _ = cv2.Rodrigues(rvec)

    proj, _ = cv2.projectPoints(CORNERS_BODY.astype(np.float64), rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    reproj_err = np.mean(np.linalg.norm(proj - corners_2d.astype(np.float64), axis=1))

    return {
        "R_cam_mav": R_cam_mav, "t_cam_mav": tvec.flatten(),
        "reprojection_error": reproj_err, "projected_corners": proj,
    }


def pnp_to_world_frame(pnp_result):
    T_cam_mav = np.eye(4)
    T_cam_mav[:3, :3] = pnp_result["R_cam_mav"]
    T_cam_mav[:3, 3] = pnp_result["t_cam_mav"]
    T_vicon_mav = np.linalg.inv(VICON2CAM_T) @ T_cam_mav
    pos_world = T_vicon_mav[:3, 3]
    q_world = R.from_matrix(T_vicon_mav[:3, :3]).as_quat()
    return pos_world, q_world


# ============================================================================
# ERROR METRICS
# ============================================================================

def angular_error_deg(q_est, q_gt):
    dot = np.clip(np.abs(np.dot(q_est, q_gt)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


# ============================================================================
# DISCOVERY
# ============================================================================

def discover_trials(data_dir):
    img_root = data_dir / "JPEGImages"
    trials = []
    for seq_dir in sorted(img_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for trial_dir in sorted(seq_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            trials.append((seq_dir.name, trial_dir.name))
    return trials


# ============================================================================
# PROCESS ONE TRIAL
# ============================================================================

def process_trial(session, sequence, model, device):
    """Process one trial, return structured dict with all data."""
    img_dir = DATA_DIR / "JPEGImages" / session / sequence
    label_dir = DATA_DIR / "labels" / session / sequence
    yolo_label_dir = DATA_DIR / "labels_yolo6d" / session / sequence

    image_files = sorted(img_dir.glob("*.jpg"), key=lambda f: int(f.stem))
    if N_IMAGES is not None:
        image_files = image_files[:N_IMAGES]

    if not image_files:
        return None

    # Pre-allocate lists
    frame_names = []
    timestamps_ns = []
    gt_pos_list, gt_q_list = [], []
    gt_cam_pos_list, gt_cam_q_list = [], []
    pnp_pos_list, pnp_q_list = [], []
    pred_corners_list, gt_corners_list, pnp_reproj_list = [], [], []
    confidence_list = []
    reproj_err_list, corner_err_list = [], []
    pos_err_list, ang_err_list = [], []
    pnp_success_list = []

    for img_path in image_files:
        name = img_path.stem
        gt_label_path = label_dir / f"{name}.txt"
        yolo_label_path = yolo_label_dir / f"{name}.txt"

        if not gt_label_path.exists() or not yolo_label_path.exists():
            continue

        # Load GT
        gt = load_gt_label(gt_label_path)
        gt_corners_px = load_yolo_label(yolo_label_path)

        # Predict
        pred_corners, confidence, err_msg = predict_corners(model, img_path, device)

        frame_names.append(name)
        timestamps_ns.append(gt["ts_cam_ns"])
        gt_pos_list.append(gt["uav_t"])
        gt_q_list.append(gt["uav_q_xyzw"])
        gt_cam_pos_list.append(gt["cam_t"])
        gt_cam_q_list.append(gt["cam_q_xyzw"])
        gt_corners_list.append(gt_corners_px)

        if pred_corners is None:
            # No detection — fill with NaN
            pnp_success_list.append(False)
            confidence_list.append(0.0)
            pred_corners_list.append(np.full((9, 2), np.nan))
            pnp_pos_list.append(np.full(3, np.nan))
            pnp_q_list.append(np.full(4, np.nan))
            pnp_reproj_list.append(np.full((9, 2), np.nan))
            reproj_err_list.append(np.nan)
            corner_err_list.append(np.nan)
            pos_err_list.append(np.nan)
            ang_err_list.append(np.nan)
            continue

        pred_corners_list.append(pred_corners)
        confidence_list.append(confidence)

        # Corner error
        corner_err = np.mean(np.linalg.norm(pred_corners[1:] - gt_corners_px[1:], axis=1))
        corner_err_list.append(corner_err)

        # PnP
        pnp = solve_pnp_camera_frame(pred_corners, use_distortion=True)

        if pnp is None:
            pnp_success_list.append(False)
            pnp_pos_list.append(np.full(3, np.nan))
            pnp_q_list.append(np.full(4, np.nan))
            pnp_reproj_list.append(np.full((9, 2), np.nan))
            reproj_err_list.append(np.nan)
            pos_err_list.append(np.nan)
            ang_err_list.append(np.nan)
            continue

        pnp_success_list.append(True)
        reproj_err_list.append(pnp["reprojection_error"])
        pnp_reproj_list.append(pnp["projected_corners"])

        pos_est, q_est = pnp_to_world_frame(pnp)
        pnp_pos_list.append(pos_est)
        pnp_q_list.append(q_est)

        pos_err = np.linalg.norm(pos_est - gt["uav_t"])
        ang_err = angular_error_deg(q_est, gt["uav_q_xyzw"])
        pos_err_list.append(pos_err)
        ang_err_list.append(ang_err)

    if not frame_names:
        return None

    # Convert timestamps to seconds from 0
    ts = np.array(timestamps_ns, dtype=np.int64)
    time_s = (ts - ts[0]) / 1e9

    return {
        "time_s": time_s,
        "frame_names": frame_names,
        "gt_pos": np.array(gt_pos_list),
        "gt_q": np.array(gt_q_list),
        "gt_cam_pos": np.array(gt_cam_pos_list),
        "gt_cam_q": np.array(gt_cam_q_list),
        "pnp_pos": np.array(pnp_pos_list),
        "pnp_q": np.array(pnp_q_list),
        "pred_corners": np.array(pred_corners_list),
        "gt_corners": np.array(gt_corners_list),
        "pnp_reproj": np.array(pnp_reproj_list),
        "confidence": np.array(confidence_list),
        "reproj_err_px": np.array(reproj_err_list),
        "corner_err_px": np.array(corner_err_list),
        "pos_err_m": np.array(pos_err_list),
        "ang_err_deg": np.array(ang_err_list),
        "pnp_success": np.array(pnp_success_list),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def make_quat_continuous(q_array):
    """Flip quaternion signs to avoid discontinuities."""
    out = q_array.copy()
    for i in range(1, len(out)):
        if not np.any(np.isnan(out[i])) and not np.any(np.isnan(out[i - 1])):
            if np.dot(out[i], out[i - 1]) < 0:
                out[i] = -out[i]
    return out


def plot_trial(session, sequence, trial_data, plots_dir):
    """Plot position, quaternion, and errors for one trial."""
    t = trial_data["time_s"]
    n = len(t)
    if n < 2:
        return

    gt_pos = trial_data["gt_pos"]
    pnp_pos = trial_data["pnp_pos"]
    gt_q = make_quat_continuous(trial_data["gt_q"])
    pnp_q = make_quat_continuous(trial_data["pnp_q"])
    ok = trial_data["pnp_success"]

    tag = f"{session}/{sequence}"

    # --- Position ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for i, lbl in enumerate(["X", "Y", "Z"]):
        axes[i].plot(t, gt_pos[:, i], "k-", lw=2, label=f"{lbl}_gt")
        axes[i].plot(t[ok], pnp_pos[ok, i], "r.", ms=3, alpha=0.7, label=f"{lbl}_pnp")
        axes[i].set_ylabel(f"{lbl} [m]")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"Position | {tag} | {n} frames")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{session}_{sequence}_position.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Quaternion ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for i, lbl in enumerate(["qx", "qy", "qz", "qw"]):
        axes[i].plot(t, gt_q[:, i], "k-", lw=2, label=f"{lbl}_gt")
        axes[i].plot(t[ok], pnp_q[ok, i], "r.", ms=3, alpha=0.7, label=f"{lbl}_pnp")
        axes[i].set_ylabel(lbl)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"Quaternion | {tag} | {n} frames")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{session}_{sequence}_quaternion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Errors ---
    pos_err = trial_data["pos_err_m"]
    ang_err = trial_data["ang_err_deg"]
    valid = ~np.isnan(pos_err)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t[valid], pos_err[valid] * 1000, "b.-", ms=3, alpha=0.7)
    axes[0].set_ylabel("Position error [mm]")
    axes[0].set_title(f"Errors | {tag} | mean pos={np.nanmean(pos_err)*1000:.1f}mm, mean ang={np.nanmean(ang_err):.2f}°")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[valid], ang_err[valid], "r.-", ms=3, alpha=0.7)
    axes[1].set_ylabel("Angular error [°]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / f"{session}_{sequence}_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Derive output folder name from model + weights
    weights_name = Path(WEIGHTS).parent.parent.name  # e.g. "exp40"
    out_dir = OUT_DIR
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"PnP EVALUATION | {MODEL_NAME} | weights: {weights_name}")
    print(f"Output: {out_dir}/")
    print("=" * 80)

    # Discover trials
    trials = discover_trials(DATA_DIR)
    print(f"Found {len(trials)} trials")

    # Load model
    print("Loading model...")
    model, device = load_model(WEIGHTS, DEVICE)
    print()

    # Process all trials
    data = {}  # data[session][sequence] = trial_data
    summary_rows = []

    for session, sequence in tqdm(trials, desc="Trials"):
        trial_data = process_trial(session, sequence, model, device)

        if trial_data is None:
            tqdm.write(f"  {session}/{sequence}: no data")
            continue

        data.setdefault(session, {})[sequence] = trial_data

        # Stats
        valid = trial_data["pnp_success"]
        n_total = len(valid)
        n_ok = valid.sum()
        pos_errs = trial_data["pos_err_m"]
        ang_errs = trial_data["ang_err_deg"]

        row = {
            "session": session, "sequence": sequence,
            "n_frames": n_total, "n_pnp_ok": int(n_ok),
            "pos_mm_mean": float(np.nanmean(pos_errs) * 1000),
            "pos_mm_median": float(np.nanmedian(pos_errs) * 1000),
            "ang_deg_mean": float(np.nanmean(ang_errs)),
            "ang_deg_median": float(np.nanmedian(ang_errs)),
        }
        summary_rows.append(row)

        # Plot
        plot_trial(session, sequence, trial_data, plots_dir)

        tqdm.write(f"  {session}/{sequence}: {n_ok}/{n_total} frames, "
                    f"pos={row['pos_mm_mean']:.1f}mm, ang={row['ang_deg_mean']:.2f}°")

    # ================================================================
    # SAVE DATA
    # ================================================================

    # Save as pickle (preserves numpy arrays)
    pickle_path = out_dir / "results.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved data dict to {pickle_path}")

    # Save summary as JSON
    all_pos = np.concatenate([d["pos_err_m"] for s in data.values() for d in s.values()])
    all_ang = np.concatenate([d["ang_err_deg"] for s in data.values() for d in s.values()])

    summary = {
        "model": MODEL_NAME,
        "weights": WEIGHTS,
        "weights_name": weights_name,
        "overall": {
            "n_trials": len(summary_rows),
            "n_frames": int(np.sum([r["n_frames"] for r in summary_rows])),
            "n_pnp_ok": int(np.sum([r["n_pnp_ok"] for r in summary_rows])),
            "pos_mm_mean": float(np.nanmean(all_pos) * 1000),
            "pos_mm_median": float(np.nanmedian(all_pos) * 1000),
            "ang_deg_mean": float(np.nanmean(all_ang)),
            "ang_deg_median": float(np.nanmedian(all_ang)),
        },
        "trials": summary_rows,
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    # ================================================================
    # PRINT SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 80)
    print(f"{'Sess/Seq':<15} {'OK/Total':>10} {'Pos mm':>10} {'Ang °':>10}")
    print("-" * 50)
    for r in summary_rows:
        print(f"{r['session']}/{r['sequence']:<10} "
              f"{r['n_pnp_ok']:>4}/{r['n_frames']:<5} "
              f"{r['pos_mm_mean']:>10.1f} {r['ang_deg_mean']:>10.2f}")
    print("-" * 50)
    print(f"{'OVERALL':<15} "
          f"{summary['overall']['n_pnp_ok']:>4}/{summary['overall']['n_frames']:<5} "
          f"{summary['overall']['pos_mm_mean']:>10.1f} {summary['overall']['ang_deg_mean']:>10.2f}")
    print("=" * 80)
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
    