#!/usr/bin/env python3
"""
fill_pose_gaps_with_trace.py

Script 2 of the pose estimation pipeline.
Uses TraceAnything to fill gaps in pose detection by propagating keypoints
along estimated trajectory fields.

Runs in the 'trace_anything' conda environment.

Usage:
    conda activate trace_anything
    python fill_pose_gaps_with_trace.py \
        --intermediate output/pose_intermediate.npz \
        --output output/ \
        --trace_anything_path /path/to/TraceAnything \
        --refine_jitter \
        --debug
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time

import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as tvf


def parse_args():
    """Parse arguments BEFORE importing TraceAnything to avoid conflicts."""
    parser = argparse.ArgumentParser(
        description="Fill pose gaps using TraceAnything trajectory fields"
    )
    parser.add_argument(
        "--intermediate",
        type=str,
        required=True,
        help="Path to intermediate .npz file from pose_estimation_with_gaps.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for final results"
    )
    parser.add_argument(
        "--trace_anything_path",
        type=str,
        default=os.path.expanduser("~/repos/dense_pose_tracker/TraceAnything"),
        help="Path to TraceAnything repository"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to TraceAnything checkpoint (default: <repo>/checkpoints/trace_anything.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TraceAnything config (default: <repo>/configs/eval.yaml)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug visualization video"
    )
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=30,
        help="Maximum gap size to attempt filling (default: 30 frames)"
    )
    parser.add_argument(
        "--max_batch_frames",
        type=int,
        default=25,
        help="Maximum frames per TraceAnything batch (default: 25, max recommended: 40)"
    )
    parser.add_argument(
        "--target_long_side",
        type=int,
        default=512,
        help="Resize frames so long side equals this (default: 512)"
    )
    # Jitter refinement options
    parser.add_argument(
        "--refine_jitter",
        action="store_true",
        help="Refine jittery detected poses using TraceAnything motion prior"
    )
    parser.add_argument(
        "--jitter_threshold",
        type=float,
        default=10.0,
        help="Pixel distance threshold to consider a keypoint as jittery (default: 10.0)"
    )
    parser.add_argument(
        "--jitter_window",
        type=int,
        default=5,
        help="Minimum frames for a jittery segment (default: 5)"
    )
    parser.add_argument(
        "--motion_mismatch_threshold",
        type=float,
        default=5.0,
        help="Threshold for motion mismatch between pose and trajectory (default: 5.0)"
    )
    return parser.parse_args()


class VideoFrameReader:
    """
    Efficient video frame reader that caches recently accessed frames.
    Reads directly from video file on demand.
    """
    
    def __init__(self, video_path: str, rotation: int = 0, cache_size: int = 100):
        self.video_path = video_path
        self.rotation = rotation
        self.cache_size = cache_size
        self.cache = {}  # frame_idx -> frame
        self.cache_order = []  # LRU tracking
        
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Adjust dimensions for rotation
        if self.rotation in [90, 270]:
            self.width, self.height = self.height, self.width
    
    def _auto_orient(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation correction to frame."""
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a single frame by index."""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        # Check cache
        if frame_idx in self.cache:
            # Move to end of LRU
            self.cache_order.remove(frame_idx)
            self.cache_order.append(frame_idx)
            return self.cache[frame_idx]
        
        # Read from video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        frame = self._auto_orient(frame)
        
        # Add to cache
        self.cache[frame_idx] = frame
        self.cache_order.append(frame_idx)
        
        # Evict if cache is full
        while len(self.cache_order) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        return frame
    
    def get_frames(self, frame_indices: List[int]) -> Dict[int, np.ndarray]:
        """Get multiple frames efficiently (sequential read when possible)."""
        frames = {}
        
        # Sort indices for efficient sequential reading
        sorted_indices = sorted(frame_indices)
        
        # Check which frames we already have cached
        to_read = [idx for idx in sorted_indices if idx not in self.cache]
        
        if to_read:
            cap = cv2.VideoCapture(self.video_path)
            
            for idx in to_read:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = self._auto_orient(frame)
                    
                    # Add to cache
                    self.cache[idx] = frame
                    self.cache_order.append(idx)
            
            cap.release()
            
            # Evict old entries
            while len(self.cache_order) > self.cache_size:
                oldest = self.cache_order.pop(0)
                if oldest in self.cache:
                    del self.cache[oldest]
        
        # Return requested frames from cache
        for idx in frame_indices:
            if idx in self.cache:
                frames[idx] = self.cache[idx]
        
        return frames
    
    def __contains__(self, frame_idx: int) -> bool:
        """Check if frame index is valid."""
        return 0 <= frame_idx < self.total_frames
    
    def __len__(self) -> int:
        return self.total_frames


def setup_trace_anything(args) -> Tuple[torch.nn.Module, torch.device]:
    """Setup TraceAnything model."""
    ta_path = Path(args.trace_anything_path)
    if not ta_path.exists():
        raise FileNotFoundError(f"TraceAnything repo not found at {ta_path}")
    
    sys.path.insert(0, str(ta_path))
    
    from omegaconf import OmegaConf
    from trace_anything.trace_anything import TraceAnything
    
    config_path = args.config or ta_path / "configs" / "eval.yaml"
    ckpt_path = args.checkpoint or ta_path / "checkpoints" / "trace_anything.pt"
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    OmegaConf.register_new_resolver("python_eval", lambda code: eval(code), replace=True)
    cfg = OmegaConf.load(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TraceAnything] Using device: {device}")
    
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[TraceAnything] GPU memory: {total_mem:.1f} GB")
    
    def to_dict(x):
        return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x
    
    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("Expected cfg.model.net or cfg.net in YAML")
    
    print(f"[TraceAnything] Building model...")
    model = TraceAnything(
        encoder_args=to_dict(net_cfg["encoder_args"]),
        decoder_args=to_dict(net_cfg["decoder_args"]),
        head_args=to_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )
    
    print(f"[TraceAnything] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if all(k.startswith("net.") for k in sd.keys()):
        sd = {k[4:]: v for k, v in sd.items()}
    
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    
    print(f"[TraceAnything] Model ready")
    return model, device


def prepare_frames_for_trace(
    frames: List[np.ndarray],
    device: torch.device,
    target_long_side: int = 512
) -> Tuple[List[Dict], Tuple[int, int], Tuple[int, int], float, float]:
    """Prepare frames for TraceAnything inference."""
    tfm = tvf.Compose([
        tvf.ToTensor(),
        tvf.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    
    views = []
    original_size = None
    processed_size = None
    scale_x = 1.0
    scale_y = 1.0
    
    for i, frame in enumerate(frames):
        if original_size is None:
            original_size = (frame.shape[0], frame.shape[1])
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        W0, H0 = pil.size
        
        if H0 > W0:
            pil = pil.transpose(Image.Transpose.ROTATE_90)
            W0, H0 = pil.size
        
        if W0 >= H0:
            new_w = target_long_side
            new_h = int(H0 * target_long_side / W0)
        else:
            new_h = target_long_side
            new_w = int(W0 * target_long_side / H0)
        
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        
        crop_w = new_w - (new_w % 16)
        crop_h = new_h - (new_h % 16)
        pil = pil.crop((0, 0, crop_w, crop_h))
        
        if processed_size is None:
            processed_size = (crop_h, crop_w)
            scale_x = crop_w / original_size[1]
            scale_y = crop_h / original_size[0]
        
        tensor = tfm(pil).unsqueeze(0).to(device)
        t = i / max(1, len(frames) - 1)
        views.append({"img": tensor, "time_step": t})
    
    return views, original_size, processed_size, scale_x, scale_y


def interpolate_keypoint_position(
    ctrl_pts: np.ndarray,
    y: int,
    x: int,
    source_t: float,
    target_t: float,
    proc_H: int,
    proc_W: int
) -> Tuple[float, float]:
    """
    Interpolate keypoint position using B-spline control points.
    
    ctrl_pts: [K, H, W, 3] where 3 = (x, y, z) in normalized coords
              x, y are in roughly [-0.5, 0.5] centered on image
    
    Returns: (delta_x, delta_y) in pixel coordinates (processed image space)
    """
    K = ctrl_pts.shape[0]
    traj = ctrl_pts[:, y, x, :]  # [K, 3]
    
    # B-spline basis (linear interpolation between control points)
    def bspline_basis(t: float, k: int, K: int) -> float:
        u = t * (K - 1)
        return max(0, 1 - abs(u - k))
    
    # Interpolate position at source and target times
    source_pos = np.zeros(3)
    target_pos = np.zeros(3)
    source_weight_sum = 0
    target_weight_sum = 0
    
    for k in range(K):
        sw = bspline_basis(source_t, k, K)
        tw = bspline_basis(target_t, k, K)
        source_pos += sw * traj[k]
        target_pos += tw * traj[k]
        source_weight_sum += sw
        target_weight_sum += tw
    
    if source_weight_sum > 0:
        source_pos /= source_weight_sum
    if target_weight_sum > 0:
        target_pos /= target_weight_sum
    
    # Delta in normalized coordinates
    delta_x_norm = target_pos[0] - source_pos[0]
    delta_y_norm = target_pos[1] - source_pos[1]
    
    # Convert from normalized [-0.5, 0.5] to pixel coordinates
    # x is normalized by width, y is normalized by height
    delta_x_pixels = delta_x_norm * proc_W
    delta_y_pixels = delta_y_norm * proc_H
    
    return delta_x_pixels, delta_y_pixels


def get_patch_motion(
    ctrl_pts: np.ndarray,
    y: int,
    x: int,
    source_t: float,
    target_t: float,
    proc_H: int,
    proc_W: int,
    patch_size: int = 3
) -> Tuple[float, float, float]:
    """
    Get average motion from a patch around the keypoint using trajectory field.
    Returns: (avg_delta_x, avg_delta_y, consistency) in pixel coordinates
    """
    half = patch_size // 2
    deltas = []
    
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            py = np.clip(y + dy, 0, proc_H - 1)
            px = np.clip(x + dx, 0, proc_W - 1)
            
            delta_x, delta_y = interpolate_keypoint_position(
                ctrl_pts, py, px, source_t, target_t, proc_H, proc_W
            )
            deltas.append((delta_x, delta_y))
    
    if not deltas:
        return 0.0, 0.0, 0.0
    
    deltas = np.array(deltas)
    avg_delta_x = np.mean(deltas[:, 0])
    avg_delta_y = np.mean(deltas[:, 1])
    
    variance = np.var(deltas[:, 0]) + np.var(deltas[:, 1])
    consistency = 1.0 / (1.0 + variance)
    
    return float(avg_delta_x), float(avg_delta_y), float(consistency)


def propagate_pose_with_trajectory(
    source_keypoints: List[List[float]],
    ctrl_pts: np.ndarray,
    ctrl_conf: np.ndarray,
    source_t: float,
    target_t: float,
    original_size: Tuple[int, int],
    processed_size: Tuple[int, int],
    scale_x: float,
    scale_y: float
) -> List[List[float]]:
    """
    Propagate all keypoints from source to target time using trajectory field.
    
    Returns keypoints with confidence based ONLY on source confidence,
    not trajectory confidence (which is about motion estimation quality,
    not keypoint validity).
    """
    proc_H, proc_W = processed_size
    
    propagated = []
    
    for kp in source_keypoints:
        orig_x, orig_y, orig_conf = kp
        
        # Skip truly invalid keypoints
        if orig_conf < 0.05:
            propagated.append([orig_x, orig_y, 0.0])
            continue
        
        # Convert original coords to processed image coords
        proc_x = orig_x * scale_x
        proc_y = orig_y * scale_y
        
        proc_x = np.clip(proc_x, 0, proc_W - 1)
        proc_y = np.clip(proc_y, 0, proc_H - 1)
        
        px = int(np.clip(round(proc_x), 0, proc_W - 1))
        py = int(np.clip(round(proc_y), 0, proc_H - 1))
        
        # Get motion delta in processed pixel coordinates
        delta_x, delta_y = interpolate_keypoint_position(
            ctrl_pts, py, px, source_t, target_t, proc_H, proc_W
        )
        
        # Apply delta
        new_proc_x = proc_x + delta_x
        new_proc_y = proc_y + delta_y
        
        new_proc_x = np.clip(new_proc_x, 0, proc_W - 1)
        new_proc_y = np.clip(new_proc_y, 0, proc_H - 1)
        
        # Convert back to original image coords
        new_orig_x = new_proc_x / scale_x
        new_orig_y = new_proc_y / scale_y
        
        # Confidence: gentle decay based on time distance only
        # The source keypoint was valid, propagation should preserve that
        time_dist = abs(target_t - source_t)
        new_conf = orig_conf * max(0.7, 1.0 - 0.15 * time_dist)
        
        propagated.append([float(new_orig_x), float(new_orig_y), float(new_conf)])
    
    return propagated


def linear_interpolate_pose(
    pose1: List[List[float]],
    pose2: List[List[float]],
    t: float
) -> List[List[float]]:
    """Linearly interpolate between two poses."""
    interpolated = []
    for kp1, kp2 in zip(pose1, pose2):
        x1, y1, c1 = kp1
        x2, y2, c2 = kp2
        
        if c1 > 0.05 and c2 > 0.05:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            c = min(c1, c2) * (1.0 - 0.3 * abs(t - 0.5))
        elif c1 > 0.05:
            x, y, c = x1, y1, c1 * (1 - t)
        elif c2 > 0.05:
            x, y, c = x2, y2, c2 * t
        else:
            x, y, c = 0, 0, 0
        
        interpolated.append([float(x), float(y), float(c)])
    
    return interpolated


def detect_jittery_segments(
    poses: Dict[str, dict],
    jitter_threshold: float = 10.0,
    window_size: int = 5
) -> List[dict]:
    """Detect segments of frames where poses are jittery."""
    frame_indices = sorted([int(k) for k in poses.keys()])
    
    if len(frame_indices) < window_size:
        return []
    
    frame_jitter = {}
    
    for i, frame_idx in enumerate(frame_indices):
        if i == 0:
            continue
        
        prev_idx = frame_indices[i - 1]
        
        if frame_idx - prev_idx != 1:
            continue
        
        curr_kps = poses[str(frame_idx)].get("keypoints", [])
        prev_kps = poses[str(prev_idx)].get("keypoints", [])
        
        if not curr_kps or not prev_kps:
            continue
        
        jitters = []
        for kp_curr, kp_prev in zip(curr_kps, prev_kps):
            if kp_curr[2] > 0.3 and kp_prev[2] > 0.3:
                dist = np.sqrt((kp_curr[0] - kp_prev[0])**2 + (kp_curr[1] - kp_prev[1])**2)
                jitters.append(dist)
        
        if jitters:
            frame_jitter[frame_idx] = jitters
    
    segments = []
    current_segment = None
    
    for frame_idx in sorted(frame_jitter.keys()):
        max_jitter = np.max(frame_jitter[frame_idx])
        is_jittery = max_jitter > jitter_threshold
        
        if is_jittery:
            if current_segment is None:
                current_segment = {
                    "start": frame_idx - 1,
                    "frames": [frame_idx - 1, frame_idx],
                    "jitters": [frame_jitter[frame_idx]]
                }
            else:
                if frame_idx - current_segment["frames"][-1] <= 2:
                    current_segment["frames"].append(frame_idx)
                    current_segment["jitters"].append(frame_jitter[frame_idx])
                else:
                    current_segment["end"] = current_segment["frames"][-1]
                    current_segment["avg_jitter"] = np.mean([np.mean(j) for j in current_segment["jitters"]])
                    if len(current_segment["frames"]) >= window_size:
                        segments.append(current_segment)
                    
                    current_segment = {
                        "start": frame_idx - 1,
                        "frames": [frame_idx - 1, frame_idx],
                        "jitters": [frame_jitter[frame_idx]]
                    }
        else:
            if current_segment is not None:
                current_segment["end"] = current_segment["frames"][-1]
                current_segment["avg_jitter"] = np.mean([np.mean(j) for j in current_segment["jitters"]])
                if len(current_segment["frames"]) >= window_size:
                    segments.append(current_segment)
                current_segment = None
    
    if current_segment is not None:
        current_segment["end"] = current_segment["frames"][-1]
        current_segment["avg_jitter"] = np.mean([np.mean(j) for j in current_segment["jitters"]])
        if len(current_segment["frames"]) >= window_size:
            segments.append(current_segment)
    
    return segments


def refine_jittery_segment(
    model: torch.nn.Module,
    device: torch.device,
    frame_reader: VideoFrameReader,
    poses: Dict[str, dict],
    segment: dict,
    max_batch_frames: int,
    target_long_side: int,
    motion_mismatch_threshold: float = 5.0
) -> Dict[str, dict]:
    """Refine poses in a jittery segment using TraceAnything motion prior."""
    refined = {}
    
    segment_frames = sorted(segment["frames"])
    start_frame = segment_frames[0]
    end_frame = segment_frames[-1]
    
    # Get frames from video
    all_indices = list(range(start_frame, end_frame + 1))
    
    # Subsample if needed
    if len(all_indices) > max_batch_frames:
        step = len(all_indices) / (max_batch_frames - 1)
        selected_indices = [all_indices[int(i * step)] for i in range(max_batch_frames - 1)]
        selected_indices.append(all_indices[-1])
        selected_indices = sorted(set(selected_indices))
    else:
        selected_indices = all_indices
    
    # Read frames
    frames_dict = frame_reader.get_frames(selected_indices)
    
    if len(frames_dict) < 3:
        return refined
    
    # Build frames list
    available_indices = sorted(frames_dict.keys())
    frames = [frames_dict[idx] for idx in available_indices]
    idx_to_batch = {idx: i for i, idx in enumerate(available_indices)}
    
    # Prepare for TraceAnything
    views, original_size, processed_size, scale_x, scale_y = prepare_frames_for_trace(
        frames, device, target_long_side
    )
    
    proc_H, proc_W = processed_size
    
    # Run inference
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
            preds = model.forward(views)
    except Exception as e:
        print(f"      TraceAnything inference failed: {e}")
        return refined
    
    total_batch = len(available_indices)
    
    # For each frame, check if pose motion matches trajectory motion
    for i, frame_idx in enumerate(available_indices):
        if i == 0:
            continue
        
        frame_key = str(frame_idx)
        prev_frame_idx = available_indices[i - 1]
        prev_frame_key = str(prev_frame_idx)
        
        if frame_key not in poses or prev_frame_key not in poses:
            continue
        
        curr_pose = poses[frame_key]
        prev_pose = poses[prev_frame_key]
        
        curr_kps = curr_pose.get("keypoints", [])
        prev_kps = prev_pose.get("keypoints", [])
        
        if not curr_kps or not prev_kps:
            continue
        
        prev_batch_idx = idx_to_batch[prev_frame_idx]
        curr_batch_idx = idx_to_batch[frame_idx]
        
        prev_t = prev_batch_idx / max(1, total_batch - 1)
        curr_t = curr_batch_idx / max(1, total_batch - 1)
        
        pred = preds[prev_batch_idx]
        ctrl_pts = pred.get("ctrl_pts3d")
        ctrl_conf = pred.get("ctrl_conf")
        
        if ctrl_pts is None:
            continue
        
        if isinstance(ctrl_pts, torch.Tensor):
            ctrl_pts = ctrl_pts.cpu().numpy()
        if ctrl_conf is not None and isinstance(ctrl_conf, torch.Tensor):
            ctrl_conf = ctrl_conf.cpu().numpy()
        else:
            ctrl_conf = np.ones(ctrl_pts.shape[:3])
        
        mismatched_keypoints = []
        
        for kp_idx, (kp_curr, kp_prev) in enumerate(zip(curr_kps, prev_kps)):
            if kp_curr[2] < 0.3 or kp_prev[2] < 0.3:
                continue
            
            # Detected motion in original image space
            detected_dx = kp_curr[0] - kp_prev[0]
            detected_dy = kp_curr[1] - kp_prev[1]
            detected_motion = np.sqrt(detected_dx**2 + detected_dy**2)
            
            # Get trajectory motion
            proc_x = kp_prev[0] * scale_x
            proc_y = kp_prev[1] * scale_y
            px = int(np.clip(round(proc_x), 0, proc_W - 1))
            py = int(np.clip(round(proc_y), 0, proc_H - 1))
            
            # Get motion in processed pixel space
            traj_dx_proc, traj_dy_proc, consistency = get_patch_motion(
                ctrl_pts, py, px, prev_t, curr_t, proc_H, proc_W, patch_size=3
            )
            
            # Convert trajectory motion back to original image space
            traj_dx_orig = traj_dx_proc / scale_x
            traj_dy_orig = traj_dy_proc / scale_y
            traj_motion = np.sqrt(traj_dx_orig**2 + traj_dy_orig**2)
            
            motion_diff = abs(detected_motion - traj_motion)
            
            if detected_motion > 3 and traj_motion > 1:
                det_dir = np.array([detected_dx, detected_dy]) / (detected_motion + 1e-6)
                traj_dir = np.array([traj_dx_orig, traj_dy_orig]) / (traj_motion + 1e-6)
                direction_agreement = np.dot(det_dir, traj_dir)
            else:
                direction_agreement = 1.0
            
            is_mismatched = (
                (motion_diff > motion_mismatch_threshold and consistency > 0.3 and detected_motion > traj_motion * 2) or
                (direction_agreement < 0.5 and detected_motion > motion_mismatch_threshold)
            )
            
            if is_mismatched:
                mismatched_keypoints.append(kp_idx)
        
        mismatch_ratio = len(mismatched_keypoints) / max(1, len([k for k in curr_kps if k[2] > 0.3]))
        
        if mismatch_ratio > 0.3:
            propagated_kps = propagate_pose_with_trajectory(
                prev_kps, ctrl_pts, ctrl_conf,
                prev_t, curr_t,
                original_size, processed_size,
                scale_x, scale_y
            )
            
            refined_kps = []
            for kp_idx, (kp_det, kp_prop) in enumerate(zip(curr_kps, propagated_kps)):
                if kp_idx in mismatched_keypoints and kp_prop[2] > 0.1:
                    refined_kps.append(kp_prop)
                else:
                    refined_kps.append(kp_det)
            
            refined[frame_key] = {
                "keypoints": refined_kps,
                "bbox": curr_pose.get("bbox", [0, 0, 100, 100]),
                "score": float(np.mean([kp[2] for kp in refined_kps if kp[2] > 0])),
                "source": "refined_jitter",
                "original_source": curr_pose.get("source", "detected"),
                "mismatched_keypoints": mismatched_keypoints
            }
    
    return refined


def fill_single_gap(
    model: torch.nn.Module,
    device: torch.device,
    frame_reader: VideoFrameReader,
    gap: dict,
    poses: Dict[str, dict],
    max_batch_frames: int,
    target_long_side: int,
    debug: bool = False
) -> Dict[str, dict]:
    """Fill a single gap using TraceAnything."""
    filled = {}
    
    start_frame = gap["start"]
    end_frame = gap["end"]
    prev_frame = gap.get("prev_frame")
    next_frame = gap.get("next_frame")
    gap_length = gap["length"]
    
    all_missing_frames = list(range(start_frame + 1, end_frame))
    
    if not all_missing_frames:
        return filled
    
    # Get frame range to read
    read_start = max(0, start_frame)
    read_end = min(len(frame_reader), end_frame + 1)
    all_indices = list(range(read_start, read_end))
    
    # Subsample if needed
    if len(all_indices) > max_batch_frames:
        step = len(all_indices) / (max_batch_frames - 1)
        selected_indices = [all_indices[int(i * step)] for i in range(max_batch_frames - 1)]
        selected_indices.append(all_indices[-1])
        # Ensure boundary frames are included
        if prev_frame is not None and prev_frame >= read_start:
            selected_indices.append(prev_frame)
        if next_frame is not None and next_frame < read_end:
            selected_indices.append(next_frame)
        selected_indices = sorted(set(selected_indices))
    else:
        selected_indices = all_indices
    
    # Read frames from video
    frames_dict = frame_reader.get_frames(selected_indices)
    
    if len(frames_dict) < 2:
        # Fall back to linear interpolation
        if prev_frame is not None and next_frame is not None:
            prev_pose = poses.get(str(prev_frame), {}).get("keypoints", [])
            next_pose = poses.get(str(next_frame), {}).get("keypoints", [])
            if prev_pose and next_pose:
                for missing_frame in all_missing_frames:
                    t = (missing_frame - start_frame) / (end_frame - start_frame)
                    interp_kps = linear_interpolate_pose(prev_pose, next_pose, t)
                    filled[str(missing_frame)] = {
                        "keypoints": interp_kps,
                        "bbox": poses.get(str(prev_frame), {}).get("bbox", [0, 0, 100, 100]),
                        "score": float(np.mean([kp[2] for kp in interp_kps if kp[2] > 0])) if any(kp[2] > 0 for kp in interp_kps) else 0.0,
                        "source": "interpolated_linear"
                    }
        return filled
    
    available_indices = sorted(frames_dict.keys())
    frames = [frames_dict[idx] for idx in available_indices]
    idx_to_batch = {idx: i for i, idx in enumerate(available_indices)}
    
    views, original_size, processed_size, scale_x, scale_y = prepare_frames_for_trace(
        frames, device, target_long_side
    )
    
    proc_H, proc_W = processed_size
    
    try:
        with torch.no_grad():
            torch.cuda.empty_cache()
            preds = model.forward(views)
    except Exception as e:
        print(f"    TraceAnything inference failed: {e}")
        return filled
    
    # Debug logging (only for first gap)
    if debug:
        print(f"\n    [DEBUG] TraceAnything output analysis:")
        print(f"    Number of predictions: {len(preds)}")
        if len(preds) > 0:
            pred = preds[0]
            print(f"    Prediction keys: {pred.keys()}")
            
            ctrl_pts = pred.get("ctrl_pts3d")
            if ctrl_pts is not None:
                if isinstance(ctrl_pts, torch.Tensor):
                    ctrl_pts_np = ctrl_pts.cpu().numpy()
                else:
                    ctrl_pts_np = ctrl_pts
                
                print(f"    ctrl_pts3d shape: {ctrl_pts_np.shape}")
                print(f"    ctrl_pts3d x range: [{ctrl_pts_np[..., 0].min():.4f}, {ctrl_pts_np[..., 0].max():.4f}]")
                print(f"    ctrl_pts3d y range: [{ctrl_pts_np[..., 1].min():.4f}, {ctrl_pts_np[..., 1].max():.4f}]")
                print(f"    ctrl_pts3d z range: [{ctrl_pts_np[..., 2].min():.4f}, {ctrl_pts_np[..., 2].max():.4f}]")
            
            ctrl_conf = pred.get("ctrl_conf")
            if ctrl_conf is not None:
                if isinstance(ctrl_conf, torch.Tensor):
                    ctrl_conf_np = ctrl_conf.cpu().numpy()
                else:
                    ctrl_conf_np = ctrl_conf
                print(f"    ctrl_conf shape: {ctrl_conf_np.shape}")
                print(f"    ctrl_conf range: [{ctrl_conf_np.min():.4f}, {ctrl_conf_np.max():.4f}]")
        
        print(f"    processed_size: {processed_size} (H, W)")
        print(f"    original_size: {original_size} (H, W)")
        print(f"    scale_x: {scale_x:.4f}, scale_y: {scale_y:.4f}")
        print(f"    [END DEBUG]\n")
    
    # Build source frames list for propagation
    source_frames = []
    if prev_frame is not None and str(prev_frame) in poses and prev_frame in idx_to_batch:
        source_frames.append(("forward", prev_frame))
    if next_frame is not None and str(next_frame) in poses and next_frame in idx_to_batch:
        source_frames.append(("backward", next_frame))
    
    if not source_frames:
        return filled
    
    total_batch_frames = len(available_indices)
    batch_filled = {}
    
    for frame_idx in available_indices:
        frame_key = str(frame_idx)
        
        if frame_key in poses:
            batch_filled[frame_idx] = poses[frame_key]["keypoints"]
            continue
        
        batch_idx = idx_to_batch[frame_idx]
        target_t = batch_idx / max(1, total_batch_frames - 1)
        
        propagations = []
        
        for direction, source_idx in source_frames:
            source_key = str(source_idx)
            source_batch_idx = idx_to_batch[source_idx]
            source_t = source_batch_idx / max(1, total_batch_frames - 1)
            
            source_pose = poses[source_key]
            source_keypoints = source_pose.get("keypoints", [])
            
            if not source_keypoints:
                continue
            
            pred = preds[source_batch_idx]
            ctrl_pts = pred.get("ctrl_pts3d")
            ctrl_conf = pred.get("ctrl_conf")
            
            if ctrl_pts is None:
                continue
            
            if isinstance(ctrl_pts, torch.Tensor):
                ctrl_pts = ctrl_pts.cpu().numpy()
            if ctrl_conf is not None and isinstance(ctrl_conf, torch.Tensor):
                ctrl_conf = ctrl_conf.cpu().numpy()
            else:
                ctrl_conf = np.ones(ctrl_pts.shape[:3])
            
            prop_kps = propagate_pose_with_trajectory(
                source_keypoints, ctrl_pts, ctrl_conf,
                source_t, target_t,
                original_size, processed_size,
                scale_x, scale_y
            )
            
            time_dist = abs(batch_idx - source_batch_idx)
            weight = 1.0 / (1.0 + time_dist * 0.1)  # Gentler distance penalty
            propagations.append((prop_kps, weight))
        
        if not propagations:
            continue
        
        num_kps = len(propagations[0][0])
        blended = []
        
        # Blending loop
        for kp_idx in range(num_kps):
            weighted_x = 0.0
            weighted_y = 0.0
            total_weight = 0.0
            max_conf = 0.0
            
            for prop_kps, weight in propagations:
                if kp_idx < len(prop_kps):
                    x, y, conf = prop_kps[kp_idx]
                    if conf > 0.05:  # Source keypoint was valid
                        weighted_x += x * weight
                        weighted_y += y * weight
                        total_weight += weight
                        max_conf = max(max_conf, conf)
            
            if total_weight > 0:
                blended.append([
                    weighted_x / total_weight,
                    weighted_y / total_weight,
                    max_conf  # Preserve the confidence from source
                ])
            else:
                # No valid propagation - this keypoint was invalid in source
                blended.append([0.0, 0.0, 0.0])
        
        batch_filled[frame_idx] = blended
        filled[frame_key] = {
            "keypoints": blended,
            "bbox": poses.get(str(prev_frame), poses.get(str(next_frame), {})).get("bbox", [0, 0, 100, 100]),
            "score": float(np.mean([kp[2] for kp in blended if kp[2] > 0])) if any(kp[2] > 0 for kp in blended) else 0.0,
            "source": "propagated_trace_anything"
        }
    
    # Second pass: interpolate frames not in batch
    all_filled_indices = sorted(batch_filled.keys())
    
    for missing_frame in all_missing_frames:
        if str(missing_frame) in filled:
            continue
        
        prev_filled = None
        next_filled = None
        
        for idx in all_filled_indices:
            if idx < missing_frame:
                prev_filled = idx
            elif idx > missing_frame and next_filled is None:
                next_filled = idx
                break
        
        if prev_frame is not None and prev_frame < missing_frame:
            if prev_filled is None or prev_frame > prev_filled:
                prev_filled = prev_frame
        if next_frame is not None and next_frame > missing_frame:
            if next_filled is None or next_frame < next_filled:
                next_filled = next_frame
        
        if prev_filled is None and next_filled is None:
            continue
        
        if prev_filled is not None:
            if prev_filled in batch_filled:
                prev_kps = batch_filled[prev_filled]
            elif str(prev_filled) in poses:
                prev_kps = poses[str(prev_filled)]["keypoints"]
            else:
                prev_kps = None
        else:
            prev_kps = None
        
        if next_filled is not None:
            if next_filled in batch_filled:
                next_kps = batch_filled[next_filled]
            elif str(next_filled) in poses:
                next_kps = poses[str(next_filled)]["keypoints"]
            else:
                next_kps = None
        else:
            next_kps = None
        
        if prev_kps is not None and next_kps is not None:
            t = (missing_frame - prev_filled) / (next_filled - prev_filled)
            interp_kps = linear_interpolate_pose(prev_kps, next_kps, t)
            source = "interpolated_between_propagated"
        elif prev_kps is not None:
            interp_kps = prev_kps
            source = "copied_from_previous"
        elif next_kps is not None:
            interp_kps = next_kps
            source = "copied_from_next"
        else:
            continue
        
        filled[str(missing_frame)] = {
            "keypoints": interp_kps,
            "bbox": poses.get(str(prev_frame), poses.get(str(next_frame), {})).get("bbox", [0, 0, 100, 100]),
            "score": float(np.mean([kp[2] for kp in interp_kps if kp[2] > 0])) if any(kp[2] > 0 for kp in interp_kps) else 0.0,
            "source": source
        }
    
    return filled


def fill_gaps_with_trace_anything(
    model: torch.nn.Module,
    device: torch.device,
    frame_reader: VideoFrameReader,
    poses: Dict[str, dict],
    gaps: List[dict],
    max_gap_frames: int,
    max_batch_frames: int,
    target_long_side: int,
    debug: bool = False
) -> Dict[str, dict]:
    """Fill all pose gaps using TraceAnything trajectory estimation."""
    filled_poses = {k: dict(v) for k, v in poses.items()}
    
    print(f"\n[GapFill] Processing {len(gaps)} gaps...")
    
    for gap_idx, gap in enumerate(gaps):
        start_frame = gap["start"]
        end_frame = gap["end"]
        gap_length = gap["length"]
        
        print(f"\n  Gap {gap_idx + 1}/{len(gaps)}: frames {start_frame + 1} to {end_frame - 1} ({gap_length} frames)")
        
        if gap_length > max_gap_frames:
            print(f"    Gap too large ({gap_length} > {max_gap_frames}), skipping")
            continue
        
        if gap_length == 0:
            continue
        
        # Only debug first gap
        filled = fill_single_gap(
            model, device, frame_reader, gap, filled_poses,
            max_batch_frames, target_long_side,
            debug=(debug and gap_idx == 0)
        )
        
        filled_poses.update(filled)
        print(f"    Filled {len(filled)} frames")
    
    return filled_poses


def refine_jittery_poses(
    model: torch.nn.Module,
    device: torch.device,
    frame_reader: VideoFrameReader,
    poses: Dict[str, dict],
    jitter_threshold: float,
    jitter_window: int,
    motion_mismatch_threshold: float,
    max_batch_frames: int,
    target_long_side: int
) -> Dict[str, dict]:
    """Detect and refine jittery pose segments using TraceAnything motion prior."""
    print(f"\n[JitterRefine] Detecting jittery segments (threshold={jitter_threshold}px, window={jitter_window})...")
    
    segments = detect_jittery_segments(poses, jitter_threshold, jitter_window)
    
    if not segments:
        print(f"[JitterRefine] No jittery segments detected")
        return poses
    
    print(f"[JitterRefine] Found {len(segments)} jittery segments:")
    for i, seg in enumerate(segments):
        print(f"  Segment {i + 1}: frames {seg['start']}-{seg['end']} ({len(seg['frames'])} frames, avg jitter={seg['avg_jitter']:.1f}px)")
    
    refined_poses = {k: dict(v) for k, v in poses.items()}
    total_refined = 0
    
    for seg_idx, segment in enumerate(segments):
        print(f"\n  Refining segment {seg_idx + 1}/{len(segments)}...")
        
        refined = refine_jittery_segment(
            model, device, frame_reader, refined_poses, segment,
            max_batch_frames, target_long_side, motion_mismatch_threshold
        )
        
        refined_poses.update(refined)
        total_refined += len(refined)
        print(f"    Refined {len(refined)} frames")
    
    print(f"\n[JitterRefine] Total refined: {total_refined} frames")
    return refined_poses


def create_debug_video(
    video_path: str,
    poses: Dict[str, dict],
    output_path: str,
    fps: float = 30.0,
    rotation: int = 0
):
    """Create debug visualization video."""
    
    def auto_oriented_frame(frame, rotation):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video: {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        return
    frame = auto_oriented_frame(frame, rotation)
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    source_colors = {
        "detected": (0, 255, 0),
        "propagated_trace_anything": (0, 165, 255),
        "interpolated_between_propagated": (255, 165, 0),
        "interpolated_linear": (255, 255, 0),
        "copied_from_previous": (128, 128, 255),
        "copied_from_next": (255, 128, 128),
        "refined_jitter": (255, 0, 255),
    }
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = auto_oriented_frame(frame, rotation)
        frame_key = str(frame_idx)
        
        if frame_key in poses:
            pose = poses[frame_key]
            keypoints = pose.get("keypoints", [])
            source = pose.get("source", "detected")
            
            color = source_colors.get(source, (255, 255, 255))
            
            if source == "detected":
                label = "Detected"
            elif source == "refined_jitter":
                label = "Refined"
            elif "propagated" in source:
                label = "Propagated"
            elif "interpolated" in source:
                label = "Interpolated"
            else:
                label = source
            
            for (i, j) in skeleton:
                if i < len(keypoints) and j < len(keypoints):
                    if keypoints[i][2] > 0.1 and keypoints[j][2] > 0.1:
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(frame, pt1, pt2, color, 2)
            
            for kp in keypoints:
                if kp[2] > 0.1:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            
            cv2.putText(frame, f"Frame {frame_idx}: {label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, f"Frame {frame_idx}: No pose", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"[Debug] Saved visualization to {output_path}")


def main():
    args = parse_args()
    
    print(f"Loading intermediate data from {args.intermediate}")
    data = np.load(args.intermediate, allow_pickle=True)
    
    metadata = data["metadata"].item()
    poses = data["poses"].item()
    gaps = list(data["gaps"])
    
    # Get video path
    video_path = metadata.get("stab_video") or metadata.get("input_video")
    if not video_path or not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    rotation = metadata.get("rotation", 0)
    
    print(f"  Video: {video_path}")
    print(f"  Rotation: {rotation}°")
    print(f"  Total frames: {metadata['total_frames']}")
    print(f"  Detected poses: {len(poses)}")
    print(f"  Gaps to fill: {len(gaps)}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frame reader
    print("\n=== Initializing video frame reader ===")
    frame_reader = VideoFrameReader(video_path, rotation=rotation, cache_size=100)
    print(f"  Video has {len(frame_reader)} frames")
    
    # Initialize TraceAnything if needed
    need_trace_anything = len(gaps) > 0 or args.refine_jitter
    
    if need_trace_anything:
        print("\n=== Initializing TraceAnything ===")
        model, device = setup_trace_anything(args)
    else:
        model, device = None, None
    
    # Fill gaps
    if len(gaps) > 0:
        print("\n=== Filling gaps with TraceAnything ===")
        filled_poses = fill_gaps_with_trace_anything(
            model, device, frame_reader, poses, gaps,
            args.max_gap_frames, args.max_batch_frames, args.target_long_side,
            debug=args.debug
        )
    else:
        print("\nNo gaps to fill")
        filled_poses = {k: dict(v) for k, v in poses.items()}
    
    # Refine jittery poses
    if args.refine_jitter:
        print("\n=== Refining jittery poses ===")
        filled_poses = refine_jittery_poses(
            model, device, frame_reader, filled_poses,
            args.jitter_threshold, args.jitter_window, args.motion_mismatch_threshold,
            args.max_batch_frames, args.target_long_side
        )
    
    # Count results
    source_counts = {}
    for p in filled_poses.values():
        source = p.get("source", "detected")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n=== Results ===")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print(f"  Total poses: {len(filled_poses)}")
    print(f"  Coverage: {len(filled_poses)}/{metadata['total_frames']} frames ({100*len(filled_poses)/metadata['total_frames']:.1f}%)")
    
    output_json = output_dir / "pose_estimations.json"
    with open(output_json, "w") as f:
        json.dump(filled_poses, f, indent=2)
    print(f"\nSaved final poses to {output_json}")
    
    if args.debug:
        video_name = Path(video_path).stem
        debug_video_path = output_dir / f"{video_name}_pose_estimation_filled.mp4"
        
        fps = metadata.get("fps", 30.0)
        
        print(f"\nCreating debug video...")
        create_debug_video(video_path, filled_poses, str(debug_video_path), fps, rotation)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()