#!/usr/bin/env python3
"""
pose_estimation_with_gaps.py

Script 1 of the pose estimation pipeline.
Uses RT-DETR for person detection and VitPose for keypoint estimation.
Identifies gaps where pose detection fails and saves intermediate data
for gap filling with TraceAnything.

Runs in the 'vitposeXformer' conda environment.

Usage:
    conda activate vitposeXformer
    python pose_estimation_with_gaps.py \
        --input_video /path/to/original.mp4 \
        --stab_video /path/to/stabilized.mp4 \
        --output output/ \
        --degrade \
        --debug
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose estimation with gap detection for TraceAnything filling"
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to original input video"
    )
    parser.add_argument(
        "--stab_video",
        type=str,
        default=None,
        help="Path to stabilized video (optional, uses input_video if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--person_detector",
        type=str,
        default="PekingU/rtdetr_r50vd_coco_o365",
        help="Person detector model name"
    )
    parser.add_argument(
        "--keypoint_detector",
        type=str,
        default="usyd-community/vitpose-base-simple",
        help="Keypoint detection model name"
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="Intentionally degrade frames to cause more detection failures (for testing)"
    )
    parser.add_argument(
        "--degrade_scale",
        type=float,
        default=0.5,
        help="Scale factor for degradation (default: 0.5 = quarter resolution)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate debug visualization video"
    )
    parser.add_argument(
        "--person_confidence",
        type=float,
        default=0.3,
        help="Minimum person detection confidence (default: 0.3)"
    )
    parser.add_argument(
        "--max_tracking_distance",
        type=float,
        default=200.0,
        help="Maximum distance (pixels) to associate detections across frames (default: 200)"
    )
    return parser.parse_args()


def get_video_rotation(video_path: str) -> int:
    """Get video rotation from metadata."""
    try:
        import subprocess
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate', '-of', 'default=nw=1:nk=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def auto_oriented_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation correction to frame."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def setup_models(device: torch.device, person_detector: str, keypoint_detector: str):
    """Initialize RT-DETR and VitPose models."""
    print(f"Loading person detection model: {person_detector}")
    person_processor = AutoProcessor.from_pretrained(person_detector, use_fast=True)
    person_model = RTDetrForObjectDetection.from_pretrained(person_detector, device_map=device)
    
    print(f"Loading keypoint detection model: {keypoint_detector}")
    pose_processor = AutoProcessor.from_pretrained(keypoint_detector, use_fast=True)
    pose_model = VitPoseForPoseEstimation.from_pretrained(keypoint_detector, device_map=device)
    
    return person_processor, person_model, pose_processor, pose_model


def degrade_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Degrade frame quality to simulate detection failures."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return degraded


def process_frame(
    person_processor,
    person_model,
    pose_processor,
    pose_model,
    input_image: Image.Image,
    person_confidence: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single frame to detect persons and estimate poses.
    
    Returns:
        xy: keypoint positions [N, 17, 2]
        scores: keypoint confidence scores [N, 17]
        boxes: person bounding boxes in VOC format [N, 4] (x1, y1, x2, y2)
    """
    # Detect persons
    inputs = person_processor(images=input_image, return_tensors="pt").to(person_model.device)
    
    with torch.no_grad():
        detections = person_model(**inputs)
    
    pp_detections = person_processor.post_process_object_detection(
        detections,
        target_sizes=torch.tensor([(input_image.height, input_image.width)]),
        threshold=person_confidence
    )[0]
    
    # Filter for person class (label 0 in COCO)
    person_mask = pp_detections["labels"] == 0
    person_boxes_voc = pp_detections["boxes"][person_mask].cpu().numpy()
    
    if len(person_boxes_voc) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format for VitPose
    person_boxes_coco = person_boxes_voc.copy()
    person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]  # width
    person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]  # height
    
    # Estimate poses
    inputs = pose_processor(input_image, boxes=[person_boxes_coco], return_tensors="pt").to(pose_model.device)
    
    with torch.no_grad():
        outputs = pose_model(**inputs)
    
    pose_results = pose_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_coco])
    image_pose_result = pose_results[0]
    
    if len(image_pose_result) == 0:
        return np.array([]), np.array([]), np.array([])
    
    xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
    scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()
    
    return xy, scores, person_boxes_voc


def get_bbox_centroid(bbox: np.ndarray) -> Tuple[float, float]:
    """Get centroid of bounding box (VOC format: x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def interactive_select_runner(video_path: str, rotation: int = 0) -> Optional[Tuple[float, float, int]]:
    """
    Interactive UI to select which runner to track.
    Returns the query point (centroid of selected runner) and the frame index.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    current_frame = 0
    query_point = None
    selected = False
    
    window_name = "Select Runner - Click to select, Enter to confirm, ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal query_point
        if event == cv2.EVENT_LBUTTONDOWN:
            query_point = (x, y)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = auto_oriented_frame(frame, rotation)
        display = frame.copy()
        
        # Draw query point if set
        if query_point:
            cv2.circle(display, (int(query_point[0]), int(query_point[1])), 10, (0, 255, 0), -1)
            cv2.circle(display, (int(query_point[0]), int(query_point[1])), 15, (0, 255, 0), 2)
        
        # Draw frame info
        cv2.putText(display, f"Frame: {current_frame}/{total_frames-1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Click on runner to track, Enter to confirm", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Arrow keys/A,D: navigate | W,S: jump 1s | ESC/Q: quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or Q
            break
        elif key == 13:  # Enter
            if query_point:
                selected = True
                break
        elif key == 83 or key == ord('d'):  # Right arrow or D
            current_frame = min(current_frame + 1, total_frames - 1)
        elif key == 81 or key == ord('a'):  # Left arrow or A
            current_frame = max(current_frame - 1, 0)
        elif key == 82 or key == ord('w'):  # Up arrow or W - jump forward 1s
            current_frame = min(current_frame + int(fps), total_frames - 1)
        elif key == 84 or key == ord('s'):  # Down arrow or S - jump backward 1s
            current_frame = max(current_frame - int(fps), 0)
    
    selection_frame = current_frame
    cap.release()
    cv2.destroyAllWindows()
    
    if selected and query_point:
        return (query_point[0], query_point[1], selection_frame)
    return None


def find_gaps(
    poses: Dict[str, dict],
    total_frames: int,
    min_gap_size: int = 1
) -> List[dict]:
    """
    Find gaps in pose detection.
    Returns list of gap info with boundaries and context.
    """
    detected_frames = sorted([int(k) for k in poses.keys()])
    gaps = []
    
    if not detected_frames:
        # Everything is a gap
        if total_frames > 0:
            gaps.append({
                "start": -1,
                "end": total_frames,
                "prev_frame": None,
                "next_frame": None,
                "length": total_frames
            })
        return gaps
    
    # Gap at the beginning
    if detected_frames[0] > 0:
        gaps.append({
            "start": -1,
            "end": detected_frames[0],
            "prev_frame": None,
            "next_frame": detected_frames[0],
            "length": detected_frames[0]
        })
    
    # Gaps in the middle
    for i in range(len(detected_frames) - 1):
        gap_start = detected_frames[i]
        gap_end = detected_frames[i + 1]
        gap_length = gap_end - gap_start - 1
        
        if gap_length >= min_gap_size:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "prev_frame": gap_start,
                "next_frame": gap_end,
                "length": gap_length
            })
    
    # Gap at the end
    if detected_frames[-1] < total_frames - 1:
        gaps.append({
            "start": detected_frames[-1],
            "end": total_frames,
            "prev_frame": detected_frames[-1],
            "next_frame": None,
            "length": total_frames - 1 - detected_frames[-1]
        })
    
    return gaps


def create_debug_video(
    video_path: str,
    poses: Dict[str, dict],
    output_path: str,
    fps: float = 30.0,
    rotation: int = 0
):
    """Create debug visualization showing detections and gaps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video for debug: {video_path}")
        return
    
    # Read first frame to get dimensions after rotation
    ret, frame = cap.read()
    if not ret:
        return
    frame = auto_oriented_frame(frame, rotation)
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # COCO skeleton connections
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
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
            
            # Color based on source
            if source == "detected":
                color = (0, 255, 0)  # Green
                label = "Detected"
            else:
                color = (0, 165, 255)  # Orange
                label = "Propagated"
            
            # Draw skeleton
            for (i, j) in skeleton:
                if i < len(keypoints) and j < len(keypoints):
                    if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
                        pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                        pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                        cv2.line(frame, pt1, pt2, color, 2)
            
            # Draw keypoints
            for kp in keypoints:
                if kp[2] > 0.3:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, color, -1)
            
            cv2.putText(frame, f"Frame {frame_idx}: {label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, f"Frame {frame_idx}: MISSING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"[Debug] Saved detection video to {output_path}")


def main():
    args = parse_args()
    
    # Setup paths
    video_path = args.stab_video if args.stab_video else args.input_video
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Setup device
    device = Accelerator().device
    print(f"Using device: {device}")
    
    # Get video rotation
    rotation = get_video_rotation(args.input_video)
    print(f"Video rotation: {rotation}°")
    
    # Interactive selection
    print("\n=== Interactive Runner Selection ===")
    selection = interactive_select_runner(video_path, rotation)
    if selection is None:
        print("No runner selected, exiting.")
        return
    query_x, query_y, selection_frame = selection
    query_point = (query_x, query_y)
    print(f"Selected query point: {query_point} at frame {selection_frame}")
    
    # Load models
    print("\n=== Loading Models ===")
    person_processor, person_model, pose_processor, pose_model = setup_models(
        device, args.person_detector, args.keypoint_detector
    )
    
    # Process video
    print("\n=== Processing Video ===")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get dimensions after rotation
    ret, frame = cap.read()
    frame = auto_oriented_frame(frame, rotation)
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    poses = {}
    last_known_centroid = query_point
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = auto_oriented_frame(frame, rotation)
        
        # Optionally degrade frame for detection
        detect_frame = degrade_frame(frame, args.degrade_scale) if args.degrade else frame
        
        # Convert to PIL for processing
        pil_image = Image.fromarray(cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB))
        
        # Process frame
        xy, scores, boxes = process_frame(
            person_processor, person_model,
            pose_processor, pose_model,
            pil_image,
            args.person_confidence
        )
        
        if len(xy) > 0 and len(boxes) > 0:
            # Find the person closest to last known centroid
            best_idx = None
            best_dist = float('inf')
            
            for i, box in enumerate(boxes):
                centroid = get_bbox_centroid(box)
                dist = distance(centroid, last_known_centroid)
                if dist < best_dist and dist < args.max_tracking_distance:
                    best_dist = dist
                    best_idx = i
            
            if best_idx is not None:
                # Store pose for the tracked person
                keypoints_with_conf = []
                for kp, score in zip(xy[best_idx], scores[best_idx]):
                    keypoints_with_conf.append([float(kp[0]), float(kp[1]), float(score)])
                
                poses[str(frame_idx)] = {
                    "keypoints": keypoints_with_conf,
                    "bbox": boxes[best_idx].tolist(),
                    "score": float(np.mean(scores[best_idx])),
                    "source": "detected"
                }
                
                # Update tracking centroid
                last_known_centroid = get_bbox_centroid(boxes[best_idx])
    
    cap.release()
    
    print(f"\nDetected poses in {len(poses)}/{total_frames} frames ({100*len(poses)/total_frames:.1f}%)")
    
    # Find gaps
    gaps = find_gaps(poses, total_frames)
    print(f"Found {len(gaps)} gaps:")
    for i, gap in enumerate(gaps[:10]):  # Show first 10
        print(f"  Gap {i}: frames {gap['start']+1} to {gap['end']-1} ({gap['length']} frames)")
    if len(gaps) > 10:
        print(f"  ... and {len(gaps) - 10} more gaps")
    
    # Save intermediate results (no frame storage - Script 2 reads from video)
    intermediate_path = output_dir / "pose_intermediate.npz"
    
    metadata = {
        "input_video": str(args.input_video),
        "stab_video": str(args.stab_video) if args.stab_video else str(args.input_video),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "rotation": rotation,
        "query_point": query_point,
        "selection_frame": selection_frame,
        "degraded": args.degrade,
    }
    
    np.savez(
        intermediate_path,
        metadata=metadata,
        poses=poses,
        gaps=gaps,
    )
    print(f"\nSaved intermediate data to {intermediate_path}")
    
    # Save initial poses JSON
    initial_json = output_dir / "pose_estimations_initial.json"
    with open(initial_json, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Saved initial poses to {initial_json}")
    
    # Create debug video if requested
    if args.debug:
        video_name = Path(video_path).stem
        debug_path = output_dir / f"{video_name}_pose_estimation_initial.mp4"
        create_debug_video(video_path, poses, str(debug_path), fps, rotation)
    
    print("\n=== Script 1 Complete ===")
    if len(gaps) > 0:
        print(f"Run Script 2 to fill {len(gaps)} gaps with TraceAnything:")
        print(f"  conda activate trace_anything")
        print(f"  python fill_pose_gaps_with_trace.py --intermediate {intermediate_path} --output {output_dir}")
    else:
        print("No gaps to fill - all frames have pose detections!")
    
    # Print jitter refinement suggestion
    print(f"\nTo also refine jittery poses, add --refine_jitter:")
    print(f"  python fill_pose_gaps_with_trace.py --intermediate {intermediate_path} --output {output_dir} --refine_jitter --debug")


if __name__ == "__main__":
    main()