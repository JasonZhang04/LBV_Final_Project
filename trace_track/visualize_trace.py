#!/usr/bin/env python3
"""
Visualize 3D flow fields from TraceAnything output.pt in 2D with color-coded direction vectors.

Usage:
    python visualize_flow.py --output output.pt --video input_video.mp4
"""

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path


def get_video_info(video_path):
    """Get video properties."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count}


def compute_frame_mapping(num_video_frames, num_pred_frames):
    """
    Compute mapping from original video frames to prediction frames.
    
    TraceAnything subsamples: if len(views) > 40, stride = len(views) // 39
    So prediction frame i corresponds to original frame i * stride.
    
    For visualization, we need the reverse: for each original frame,
    find the nearest prediction frame.
    
    Returns:
        pred_indices: array of length num_video_frames, mapping each video frame 
                      to its corresponding prediction index
        stride: the subsampling stride used
    """
    if num_video_frames > 40:
        stride = max(1, num_video_frames // 39)
    else:
        stride = 1
    
    # Original frames that were actually processed
    sampled_orig_frames = list(range(0, num_video_frames, stride))[:num_pred_frames]
    
    # For each original video frame, find nearest prediction
    pred_indices = []
    for orig_idx in range(num_video_frames):
        # Find closest sampled frame
        best_pred_idx = 0
        best_dist = abs(orig_idx - sampled_orig_frames[0])
        for pred_idx, sampled_orig in enumerate(sampled_orig_frames):
            dist = abs(orig_idx - sampled_orig)
            if dist < best_dist:
                best_dist = dist
                best_pred_idx = pred_idx
        pred_indices.append(best_pred_idx)
    
    return np.array(pred_indices), stride


def flow_to_color_wheel(flow_xy, max_flow=None):
    """
    Convert 2D optical flow to color using the standard flow color wheel.
    """
    fx, fy = flow_xy[..., 0], flow_xy[..., 1]
    
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    
    if max_flow is None:
        max_flow = magnitude.max() + 1e-8
    magnitude_norm = np.clip(magnitude / max_flow, 0, 1)
    
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.float32)
    sat = np.ones_like(hue) * 255
    val = (magnitude_norm * 255).astype(np.float32)
    
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def flow_to_color_wheel_with_depth(flow_3d, max_flow=None):
    """
    Convert 3D flow to color, encoding XY direction in hue and Z in brightness.
    """
    fx, fy, fz = flow_3d[..., 0], flow_3d[..., 1], flow_3d[..., 2]
    
    mag_xy = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)
    magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
    
    if max_flow is None:
        max_flow = magnitude.max() + 1e-8
    
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.float32)
    sat = np.clip(mag_xy / (magnitude + 1e-8), 0, 1) * 255
    val = np.clip(magnitude / max_flow, 0, 1) * 255
    
    hsv = np.stack([hue, sat.astype(np.float32), val.astype(np.float32)], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def create_color_wheel_legend(size=100):
    """Create a color wheel legend for flow visualization."""
    y, x = np.mgrid[-size:size+1, -size:size+1].astype(np.float32)
    
    angle = np.arctan2(y, x)
    magnitude = np.sqrt(x**2 + y**2)
    magnitude_norm = np.clip(magnitude / size, 0, 1)
    
    mask = magnitude <= size
    
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.float32)
    sat = np.ones_like(hue) * 255
    val = (magnitude_norm * 255 * mask).astype(np.float32)
    
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb[~mask] = 255
    
    return rgb


def draw_flow_arrows(img, flow_xy, step=20, scale=1.0, color=(255, 255, 255), thickness=1):
    """Draw flow arrows on image."""
    img_out = img.copy()
    H, W = flow_xy.shape[:2]
    
    for y in range(step//2, H, step):
        for x in range(step//2, W, step):
            fx, fy = flow_xy[y, x, 0], flow_xy[y, x, 1]
            mag = np.sqrt(fx**2 + fy**2)
            
            if mag > 0.01:
                end_x = int(x + fx * scale)
                end_y = int(y + fy * scale)
                cv2.arrowedLine(img_out, (x, y), (end_x, end_y), color, thickness, tipLength=0.3)
    
    return img_out


def compute_trajectory_flow(ctrl_pts3d, k_start=0, k_end=None):
    """Compute flow from trajectory control points."""
    K = ctrl_pts3d.shape[0]
    
    if k_end is None:
        k_end = min(k_start + 1, K - 1)
    
    k_start = min(k_start, K - 1)
    k_end = min(k_end, K - 1)
    
    flow_3d = ctrl_pts3d[k_end] - ctrl_pts3d[k_start]
    
    return flow_3d


def compute_global_flow_stats(preds, k_delta=1):
    """Compute global flow statistics for normalization."""
    max_flow_3d = 0
    max_flow_xy = 0
    max_flow_z = 0
    
    for i in range(len(preds)):
        ctrl_pts3d = preds[i]['ctrl_pts3d']
        if isinstance(ctrl_pts3d, torch.Tensor):
            ctrl_pts3d = ctrl_pts3d.numpy()
        
        K = ctrl_pts3d.shape[0]
        flow_3d = compute_trajectory_flow(ctrl_pts3d, k_start=0, k_end=min(k_delta, K-1))
        
        mag_3d = np.sqrt((flow_3d**2).sum(axis=-1)).max()
        mag_xy = np.sqrt((flow_3d[..., :2]**2).sum(axis=-1)).max()
        mag_z = np.abs(flow_3d[..., 2]).max()
        
        max_flow_3d = max(max_flow_3d, mag_3d)
        max_flow_xy = max(max_flow_xy, mag_xy)
        max_flow_z = max(max_flow_z, mag_z)
    
    return {
        'max_3d': max_flow_3d + 1e-8,
        'max_xy': max_flow_xy + 1e-8,
        'max_z': max_flow_z + 1e-8
    }


def render_flow_frame(flow_3d, target_W, target_H, mode, flow_stats, legend=None, 
                      show_arrows=False, arrow_step=25, arrow_scale=2.0):
    """Render a single flow visualization frame."""
    flow_H, flow_W = flow_3d.shape[:2]
    
    # Resize flow to target resolution
    flow_3d_resized = cv2.resize(flow_3d, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
    
    # Scale flow vectors to account for resolution change
    scale_x = target_W / flow_W
    scale_y = target_H / flow_H
    flow_3d_resized[..., 0] *= scale_x
    flow_3d_resized[..., 1] *= scale_y
    
    # Scaled max flows
    max_flow_xy_scaled = flow_stats['max_xy'] * scale_x
    max_flow_3d_scaled = flow_stats['max_3d'] * max(scale_x, scale_y)
    
    if mode == 'flow_color':
        vis = flow_to_color_wheel(flow_3d_resized[..., :2], max_flow=max_flow_xy_scaled)
        
    elif mode == 'flow_color_3d':
        vis = flow_to_color_wheel_with_depth(flow_3d_resized, max_flow=max_flow_3d_scaled)
        
    elif mode == 'depth_change':
        z_flow = flow_3d_resized[..., 2]
        z_normalized = z_flow / (flow_stats['max_z'] * scale_y)
        z_normalized = np.clip(z_normalized, -1, 1)
        
        vis = np.zeros((target_H, target_W, 3), dtype=np.uint8)
        vis[..., 0] = np.clip((z_normalized + 1) / 2 * 255, 0, 255).astype(np.uint8)
        vis[..., 2] = np.clip((1 - z_normalized) / 2 * 255, 0, 255).astype(np.uint8)
        vis[..., 1] = np.clip((1 - np.abs(z_normalized)) * 255, 0, 255).astype(np.uint8)
        
    elif mode == 'magnitude':
        magnitude = np.sqrt((flow_3d_resized**2).sum(axis=-1))
        mag_normalized = np.clip(magnitude / max_flow_3d_scaled, 0, 1)
        vis = cv2.applyColorMap((mag_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if show_arrows:
        vis = draw_flow_arrows(
            vis, 
            flow_3d_resized[..., :2], 
            step=arrow_step, 
            scale=arrow_scale,
            color=(255, 255, 255),
            thickness=1
        )
    
    if legend is not None and mode in ['flow_color', 'flow_color_3d']:
        lh, lw = legend.shape[:2]
        margin = 10
        if target_W > lw + margin * 2 and target_H > lh + margin * 2:
            vis[margin:margin+lh, target_W-margin-lw:target_W-margin] = legend
    
    return vis


def create_visualization(
    output_path, 
    video_path, 
    save_path,
    mode='flow_color',
    show_arrows=False,
    arrow_step=30,
    arrow_scale=3.0,
    side_by_side=True,
    k_delta=1
):
    """
    Create flow visualization video with proper frame mapping.
    """
    # Load predictions
    print(f"Loading TraceAnything output from {output_path}...")
    data = torch.load(output_path, map_location='cpu', weights_only=False)
    preds = data['preds']
    num_pred_frames = len(preds)
    print(f"  Found {num_pred_frames} predicted frames")
    
    # Get video info
    print(f"Getting video info from {video_path}...")
    video_info = get_video_info(video_path)
    fps = video_info['fps']
    target_W, target_H = video_info['width'], video_info['height']
    num_video_frames = video_info['frame_count']
    print(f"  Resolution: {target_W}x{target_H}, FPS: {fps}, Frames: {num_video_frames}")
    
    # Compute frame mapping (video frame -> prediction frame)
    pred_indices, stride = compute_frame_mapping(num_video_frames, num_pred_frames)
    print(f"  Subsampling stride: {stride}")
    print(f"  Video frames: {num_video_frames} -> Prediction frames: {num_pred_frames}")
    
    # Compute global flow statistics
    print("Computing global flow statistics...")
    flow_stats = compute_global_flow_stats(preds, k_delta=k_delta)
    print(f"  Max flow - 3D: {flow_stats['max_3d']:.3f}, XY: {flow_stats['max_xy']:.3f}, Z: {flow_stats['max_z']:.3f}")
    
    # Precompute all flows (for the ~40 prediction frames, this is fine)
    print("Precomputing flow fields...")
    all_flows = []
    for i in range(num_pred_frames):
        ctrl_pts3d = preds[i]['ctrl_pts3d']
        if isinstance(ctrl_pts3d, torch.Tensor):
            ctrl_pts3d = ctrl_pts3d.numpy()
        K = ctrl_pts3d.shape[0]
        flow_3d = compute_trajectory_flow(ctrl_pts3d, k_start=0, k_end=min(k_delta, K-1))
        all_flows.append(flow_3d)
    
    # Create color wheel legend
    legend_size = min(target_H // 6, 80)
    legend = create_color_wheel_legend(legend_size)
    
    # Setup video reader and writer
    cap = cv2.VideoCapture(video_path)
    
    if side_by_side:
        out_W = target_W * 2
    else:
        out_W = target_W
    out_H = target_H
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (out_W, out_H))
    
    print(f"Generating visualization ({mode})...")
    
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        orig_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Map this video frame to the correct prediction frame
        pred_idx = pred_indices[frame_idx]
        flow_3d = all_flows[pred_idx]
        
        # Render visualization
        vis = render_flow_frame(
            flow_3d, target_W, target_H, mode, flow_stats, legend,
            show_arrows=show_arrows, arrow_step=arrow_step, arrow_scale=arrow_scale
        )
        
        # Add text overlay
        cv2.putText(vis, f'Frame {frame_idx} (pred {pred_idx})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, mode, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine with original if side by side
        if side_by_side:
            combined = np.concatenate([orig_frame, vis], axis=1)
        else:
            combined = vis
        
        # Write frame
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{num_video_frames} frames")
    
    cap.release()
    writer.release()
    print(f"Saved to {save_path}")


def create_all_visualizations(output_path, video_path, output_dir, k_delta=1):
    """Create all visualization modes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modes = ['flow_color', 'flow_color_3d', 'depth_change', 'magnitude']
    
    for mode in modes:
        save_path = output_dir / f"flow_{mode}.mp4"
        print(f"\n{'='*50}")
        print(f"Creating {mode} visualization...")
        print('='*50)
        
        create_visualization(
            output_path=output_path,
            video_path=video_path,
            save_path=str(save_path),
            mode=mode,
            side_by_side=True,
            k_delta=k_delta
        )
    
    # Also create one with arrows
    save_path = output_dir / "flow_color_arrows.mp4"
    print(f"\n{'='*50}")
    print("Creating flow_color with arrows...")
    print('='*50)
    
    create_visualization(
        output_path=output_path,
        video_path=video_path,
        save_path=str(save_path),
        mode='flow_color',
        show_arrows=True,
        arrow_step=25,
        arrow_scale=2.0,
        side_by_side=True,
        k_delta=k_delta
    )
    
    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 3D flow fields from TraceAnything',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Visualization modes:
  flow_color    - Standard optical flow color wheel (XY direction)
  flow_color_3d - Color wheel with Z component affecting saturation
  depth_change  - Blue=toward camera, Red=away from camera
  magnitude     - Flow magnitude heatmap

Examples:
  python visualize_flow.py --output output.pt --video input.mp4
  python visualize_flow.py --output output.pt --video input.mp4 --all
  python visualize_flow.py --output output.pt --video input.mp4 --mode depth_change
  python visualize_flow.py --output output.pt --video input.mp4 --mode flow_color --arrows
        """
    )
    parser.add_argument('--output', type=str, required=True, 
                       help='Path to TraceAnything output.pt file')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to original input video')
    parser.add_argument('--save', type=str, default='flow_visualization.mp4',
                       help='Path to save output video')
    parser.add_argument('--mode', type=str, default='flow_color',
                       choices=['flow_color', 'flow_color_3d', 'depth_change', 'magnitude'],
                       help='Visualization mode')
    parser.add_argument('--arrows', action='store_true',
                       help='Overlay flow arrows')
    parser.add_argument('--arrow-step', type=int, default=25,
                       help='Spacing between arrows')
    parser.add_argument('--arrow-scale', type=float, default=2.0,
                       help='Arrow length multiplier')
    parser.add_argument('--no-side-by-side', action='store_true',
                       help='Show only visualization (no original video)')
    parser.add_argument('--k-delta', type=int, default=1,
                       help='Control point index difference for flow computation')
    parser.add_argument('--all', action='store_true',
                       help='Generate all visualization modes')
    parser.add_argument('--output-dir', type=str, default='flow_visualizations',
                       help='Output directory for --all mode')
    
    args = parser.parse_args()
    
    if args.all:
        create_all_visualizations(args.output, args.video, args.output_dir, k_delta=args.k_delta)
    else:
        create_visualization(
            output_path=args.output,
            video_path=args.video,
            save_path=args.save,
            mode=args.mode,
            show_arrows=args.arrows,
            arrow_step=args.arrow_step,
            arrow_scale=args.arrow_scale,
            side_by_side=not args.no_side_by_side,
            k_delta=args.k_delta
        )


if __name__ == '__main__':
    main()