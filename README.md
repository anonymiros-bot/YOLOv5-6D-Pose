# YOLOv5-6D for UAV Pose Estimation

Fork of [YOLOv5-6D-Pose](https://github.com/cviviers/YOLOv5-6D-Pose) adapted for vision-based UAV pose estimation.

Accompanies the paper *"Towards Agile Vision-Based Multi-UAV Flight: Revisiting State Estimation"* (submitted to IROS 2026).

## Pipeline

```
image → YOLOv5-6D → 9 keypoints (pixels) → PnP → T_cam_uav → world pose
```

1. **YOLOv5-6D** detects 9 keypoints (centroid + 8 bounding box corners) in image coordinates
2. **PnP** (`solvePnP` + Levenberg-Marquardt refinement) solves for camera-to-UAV transform using known 3D model
3. **World transform** converts to world frame using camera pose (ground truth or estimated)

Output per frame: `pnp_pos` (3D position) and `pnp_q` (orientation quaternion, xyzw).

These outputs feed into the state estimation benchmark — the estimators fuse noisy `pnp_pos` and `pnp_q` measurements to produce filtered state estimates.

## Added Scripts

| Script | Dataset | Sequences |
|--------|---------|-----------|
| `pnp_eval_all_unreal_flying.py` | Unreal (simulated) | 140 flying trajectories |
| `pnp_eval_all_unreal_step.py` | Unreal (simulated) | 35 step response |
| `pnp_eval_all_real.py` | Phantom 4 / Mavic 2 | Real-world with MoCap GT |

Each script:
- Loads trained YOLOv5-6D weights
- Runs detection + PnP on all frames in all sequences
- Transforms results to world frame
- Saves JSON with original fields + `pnp_pos`, `pnp_q` filled in

### Usage

```bash
# 1. Edit paths at top of script:
#    WEIGHTS   — path to trained .pt file
#    SEQ_ROOT  — dataset root directory
#    OUT_DIR   — where to save results

# 2. Run:
python pnp_eval_all_unreal_flying.py
```

Output: `pnp_results.json` — one entry per sequence with per-frame pose estimates.

## Notebooks

Interactive versions for single-sequence testing and visualization:

| Notebook | Purpose |
|----------|---------|
| `pose_estimation_seq_unreal_flying.ipynb` | Test on random Unreal flying sequence |
| `pose_estimation_seq_unreal_step.ipynb` | Test on step response sequence |
| `pose_estimation_seq_phantom4.ipynb` | Test on Phantom 4 real-world |
| `pose_estimation_seq_mavic2.ipynb` | Test on Mavic 2 real-world |

Useful for debugging detection failures and visualizing PnP results.

## Directory Structure

### Unreal (simulated)

```
data/sequences/flying/
├── flying/{k}/sequence.json          # GT: pos, quat, cam_pos, cam_q, ...
└── trajectory_{k}_{suffix}/drone/images/*.jpg
```

Camera: simulated pinhole, `fx=fy=960`, `cx=960`, `cy=540`, no distortion.

### Phantom 4 / Mavic 2 (real-world)

```
data/
├── JPEGImages/{session}/{sequence}/*.jpg
├── labels/{session}/{sequence}/*.txt        # MoCap GT (cam + UAV poses)
└── labels_yolo6d/{session}/{sequence}/*.txt # YOLO keypoint labels
```

Camera intrinsics and distortion coefficients defined in script headers.

## 3D Model

Keypoints are 8 corners of the UAV bounding box + centroid:

```
CORNERS_BODY = [
    centroid,
    (min_x, min_y, min_z), (min_x, min_y, max_z),
    (min_x, max_y, min_z), (min_x, max_y, max_z),
    (max_x, min_y, min_z), (max_x, min_y, max_z),
    (max_x, max_y, min_z), (max_x, max_y, max_z),
]
```

Dimensions extracted from mesh files (`.ply`). PnP uses the 8 corners; centroid is predicted but not used in pose solving.

## Output Format

Each sequence in the output JSON:

```json
{
  "0": {
    "t": [...],           // timestamps
    "pos": [...],         // GT position (N×3)
    "quat": [...],        // GT quaternion xyzw (N×4)
    "pnp_pos": [...],     // estimated position (N×3), NaN if detection failed
    "pnp_q": [...],       // estimated quaternion (N×4), NaN if detection failed
    "image_dir": "...",   // path to images
    ...
  }
}
```

NaN values indicate frames where detection or PnP failed.

## Weights

Trained weights: [Google Drive](https://drive.google.com/drive/folders/1nZsnAD0rCxkVpD7aDKQ9ucQ5p88Euxij?usp=drive_link)

```
runs/train/
├── unreal/weights/best.pt    # Simulated quadrotor
├── phantom4/weights/best.pt  # DJI Phantom 4
└── mavic2/weights/best.pt    # DJI Mavic 2
```

Trained on synthetic (Unreal Engine) and real images with YOLO keypoint labels.

## Setup & Training

See the [original YOLOv5-6D-Pose README](https://github.com/cviviers/YOLOv5-6D-Pose) for:
- Installation & dependencies
- Training on custom datasets
- Inference options

Training configs for UAV datasets: `configs/shiu/` (Phantom 4, Mavic 2).

## Citation

```bibtex
@article{TODO,
  title={Towards Agile Vision-Based Multi-UAV Flight: Revisiting State Estimation},
  author={...},
  journal={submitted to IROS 2026},
  year={2026}
}
```