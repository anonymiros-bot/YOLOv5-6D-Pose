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

These outputs feed into the [state estimation benchmark](https://github.com/anonymiros-bot/tilt-detection-and-estimation).

## Evaluation Scripts

| Script | Dataset | Sequences |
|--------|---------|-----------|
| `pnp_eval_all_unreal_flying.py` | Unreal (simulated) | 140 flying trajectories |
| `pnp_eval_all_unreal_step.py` | Unreal (simulated) | 35 step response |
| `pnp_eval_all_real_mavic2.py` | Mavic 2 (real-world) | 61 sequences, MoCap GT |
| `pnp_eval_all_real_phantom4.py` | Phantom 4 (real-world) | 68 sequences, MoCap GT |

Each script loads weights, runs detection + PnP on all frames, transforms to world frame, and saves results to `estimation_data/`. All paths are relative to the repo root — no manual editing needed.

### Usage
```bash
# After setup (see below), just run:
python pnp_eval_all_unreal_flying.py
python pnp_eval_all_unreal_step.py
python pnp_eval_all_real_mavic2.py
python pnp_eval_all_real_phantom4.py
```

Output JSONs are saved in `estimation_data/` and can be used directly with the [estimator repo](https://github.com/anonymiros-bot/tilt-detection-and-estimation).

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
```bash
./setup_data.sh    # ~70 GB, includes all flight sequences
```
```
data/sequences/flying/
├── flying/{k}/sequence.json                    # GT: pos, quat, cam_pos, cam_q, ...
└── trajectory_{k}_{suffix}/drone/images/*.jpg

data/sequences/acc_step/
├── acc_step/{k}/sequence.json                  # GT for step response sequences
└── step_a{level}_cam{id}_{k}/drone/images/*.jpg
```

Camera: simulated pinhole, `fx=fy=960`, `cx=960`, `cy=540`, no distortion.

### Phantom 4 / Mavic 2 (real-world, from [MAV6D](https://github.com/WindyLab/MAV6D))

Included in the same download as above (`setup_data.sh`).
```
data/data/{mavic2,phantom4}/
├── JPEGImages/{session}/{sequence}/*.jpg
├── labels/{session}/{sequence}/*.txt        # MoCap GT (cam + UAV poses)
└── labels_yolo6d/{session}/{sequence}/*.txt # YOLO keypoint labels
```

Camera intrinsics and distortion coefficients defined in script headers.

### Training data (Unreal)
```bash
./setup_train_data.sh    # ~15 GB
```
```
data/unreal/
├── JPEGImages/          # ~13k training images
├── labels/              # YOLO keypoint labels
├── mesh/                # 3D mesh files
├── train.txt
├── test.txt
└── validation.txt
```

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

All evaluation scripts save results to `estimation_data/`. Each sequence in the output JSON:
```json
{
  "0": {
    "t": [...],           # timestamps
    "pos": [...],         # GT position (N×3)
    "quat": [...],        # GT quaternion xyzw (N×4)
    "pnp_pos": [...],     # estimated position (N×3), NaN if detection failed
    "pnp_q": [...],       # estimated quaternion (N×4), NaN if detection failed
    "image_dir": "...",   # path to images
    ...
  }
}
```

## Weights

Download via setup script or manually:
```bash
./setup_weights.sh
```

[Manual download](https://drive.google.com/file/d/1EL4UmK5QDfc8CrBhunN8P-xHqFs4eJvv/view?usp=drive_link) (~2.7 GB) — place `runs.tar.gz` in repo root and re-run the script.
```
runs/train/
├── unreal/weights/best.pt    # Simulated quadrotor
├── phantom4/weights/best.pt  # DJI Phantom 4
└── mavic2/weights/best.pt    # DJI Mavic 2
```

Trained on synthetic (Unreal Engine) and real-world ([MAV6D](https://github.com/WindyLab/MAV6D), Zheng et al.) images with YOLO keypoint labels.

## Setup
```bash
pip install -r requirements.txt
cd utils && python setup.py build_ext --inplace && cd ..
```

See the [original YOLOv5-6D-Pose repo](https://github.com/cviviers/YOLOv5-6D-Pose) for full documentation on installation, training, and inference.

### Training config

Training config for the Unreal dataset: `configs/unreal.yaml`

## Data Loader

`load_dataset.py` provides unified access to all datasets:
```python
from load_dataset import load_dataset

data = load_dataset("unreal_flying")  # or "unreal_step", "mavic2", "phantom4"
seq = data["42"]
# seq["t"], seq["pos"], seq["quat"], seq["pnp_pos"], seq["images"], ...
```

See the [project page](https://bit.ly/4u4QXjz) for full API documentation.

## Citation
```bibtex
@article{TODO,
  title={Towards Agile Vision-Based Multi-UAV Flight: Revisiting State Estimation},
  author={...},
  journal={submitted to IROS 2026},
  year={2026}
}
```