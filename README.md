# LBV Final Project – Natural Video Flickering Notebook

## Context

Jason is responsible for the natural videos section of a Learning Based Vision final project. The goal is to demonstrate that temporal flickering exists in video pose estimation, quantify it, and show a simple mitigation. The notebook needs to run end-to-end on Google Colab. The project is exploratory — good demos and clear metrics matter more than rigor.

---

## Model Recommendation: YOLO11n-pose (Ultralytics)

**Why YOLO11 Pose first:**
- Single install: `pip install ultralytics` — zero Colab configuration friction
- Per-frame inference by default — this is exactly what causes flickering, making it the ideal tool to demonstrate the problem
- Returns structured keypoint arrays (17 COCO joints × (x, y, confidence)) — trivial to compute frame-to-frame jitter
- Multiple model sizes (nano → xlarge) so you can start fast on Colab's T4 GPU
- Well-documented with many examples; easy to add tracking later for comparison

**Why not the others for the first pass:**
- Video Swin Transformer (with YOLOv8 backbone): complex multi-component setup, slow on Colab
- DCPose: temporal-aware and reduces flickering — better as a *comparison* model, not the one that demonstrates the problem
- PoseTrack, Human3.6M, AthletePose3D, MPII: these are **datasets**, not models

---

## Dataset: UCF101 via HuggingFace (`flwrlabs/ucf101`)

**Why UCF101:**
- Freely streamable from HuggingFace with `datasets` library — no registration, no manual download
- 13,320 short action clips, many sports/athletic categories (basketball, tennis, soccer, etc.)
- Realistic camera motion, varied lighting, multiple people — all factors that amplify flickering
- Can stream just 10–20 clips for a demo without downloading the full dataset

**Secondary option:** AthletePose3D (GitHub, free) for high-speed athletic movements if more challenging clips are needed.

---

## Notebook Plan: `natural_video_flickering.ipynb`

### Section 0 – Setup
```
pip install ultralytics datasets opencv-python-headless matplotlib scipy
```
Also configure GPU detection cell.

### Section 1 – Load Video Clips
- Stream UCF101 sports subset from HuggingFace (`flwrlabs/ucf101`)
- Filter to athletic action classes (e.g., `Basketball`, `TennisSwing`, `Skiing`)
- Extract 2–3 short clips (~150 frames each) as MP4 files in Colab's `/tmp/`

### Section 2 – Per-Frame YOLO11 Pose Inference
- Load `YOLO("yolo11n-pose.pt")` (auto-downloads ~6MB checkpoint)
- Run inference frame-by-frame (not using YOLO's built-in video mode) to make flickering more visible
- Store raw keypoint arrays: shape `[num_frames, 17, 3]` (x, y, conf) per person

### Section 3 – Visualize Raw Predictions
- Render skeleton overlays on a sequence of frames (every 5th frame as a grid)
- Create a short output GIF/video showing frame-to-frame prediction changes
- This is the primary **demo** showing flickering exists

### Section 4 – Quantify Flickering
- Compute per-keypoint frame-to-frame L2 distance: `jitter[t] = ||kp[t] - kp[t-1]||`
- Aggregate: mean jitter per joint across the clip
- Plot: time-series of jitter for 3–4 key joints (wrist, knee, hip) — shows the noise signal clearly
- Summary metric: **Temporal Instability Score** = mean frame-to-frame displacement (pixels)

### Section 5 – Temporal Smoothing Baselines
Apply two simple smoothing strategies to the raw keypoint trajectories:
1. **Gaussian smoothing** (`scipy.ndimage.gaussian_filter1d`, σ=2): smooth per-joint x/y over time
2. **Moving average** (window=5 frames): simple rolling mean

Re-render skeleton overlays on the same clip with smoothed keypoints.

### Section 6 – Comparison & Results
- Side-by-side GIF: raw vs Gaussian-smoothed vs moving-average
- Bar chart: Temporal Instability Score for each condition
- Brief observation text cell noting the flickering reduction and the lag/sharpness tradeoff

---

## File to Create
- `LBV Final Project/natural_video_flickering.ipynb` — single self-contained Colab notebook

---

## Verification
1. Open notebook in Colab (Runtime → Run All)
2. Confirm UCF101 clips load and extract without errors
3. Confirm YOLO11n-pose inference runs on GPU and produces keypoint arrays
4. Confirm jitter plot shows visible high-frequency noise in raw predictions
5. Confirm smoothed video clearly shows reduced jitter
6. Confirm bar chart shows lower Temporal Instability Score for smoothed conditions
