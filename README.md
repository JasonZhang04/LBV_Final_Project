# LBV Final Project – Natural Video Flickering

Demonstrates temporal flickering in video pose estimation, quantifies it, and evaluates simple mitigations. Runs end-to-end on Google Colab.

---

## Dataset: Penn Action

2,326 natural sports video sequences (tennis, baseball, golf, etc.) with per-frame 2D joint ground-truth annotations (13 joints, MATLAB format) and visibility flags. Clips are 640×480 RGB.

---

## Model: YOLO11n-Pose (Ultralytics)

Pre-trained model running frame-by-frame inference to extract 17 COCO keypoints per person. Person identity is maintained across frames using bounding-box IoU tracking.

---

## Pipeline

1. **Load videos** — Download Penn Action sports clips and extract RGB frames with ground-truth joint annotations.
2. **Per-frame inference** — Run YOLO11n-Pose independently on each frame; track person identity via IoU to avoid detection switches.
3. **Quantify flickering** — Compute per-frame jitter and aggregate as **Temporal Instability Score (TIS)**: mean frame-to-frame L2 keypoint displacement (pixels/frame). Distal joints (wrists, ankles) show higher instability than proximal joints.
4. **Visualize** — Render skeleton overlays showing frame-to-frame jitter.
5. **Temporal smoothing** — Apply Gaussian filter (σ=2) and moving average (5-frame window) along the temporal axis.
6. **Compare** — Plot jitter trajectories and TIS before/after smoothing; generate side-by-side comparison videos and export to Google Drive.

---

## Key Finding

Smoothing reduces TIS by 40–60%, but introduces temporal lag on fast motion — a tradeoff between jitter suppression and responsiveness.
