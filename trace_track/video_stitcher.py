#!/usr/bin/env python3
"""
Video Stitcher - Combines 3 videos side by side with speed control
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def get_video_properties(cap):
    """Get video properties from a capture object."""
    return {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }


def add_label(frame, text, position='top-left', color=(0, 0, 255), 
              font_scale=1.0, thickness=2):
    """Add a text label to a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    padding = 10
    if position == 'top-left':
        x, y = padding, text_size[1] + padding
    elif position == 'top-center':
        x = (frame.shape[1] - text_size[0]) // 2
        y = text_size[1] + padding
    elif position == 'top-right':
        x = frame.shape[1] - text_size[0] - padding
        y = text_size[1] + padding
    
    # Draw background rectangle for better visibility
    cv2.rectangle(frame, 
                  (x - 5, y - text_size[1] - 5),
                  (x + text_size[0] + 5, y + 5),
                  (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame


def resize_frame(frame, target_width, target_height):
    """Resize frame to target dimensions."""
    return cv2.resize(frame, (target_width, target_height), 
                      interpolation=cv2.INTER_LINEAR)


def stitch_videos(video_paths, output_path, speed=1.0, labels=None, 
                  target_height=480, add_border=True, border_width=2):
    """
    Stitch 3 videos side by side and save to disk.
    
    Args:
        video_paths: List of 3 video file paths
        output_path: Output video file path
        speed: Playback speed multiplier (0.5 = half speed, 2.0 = double speed)
        labels: Optional list of 3 labels for each video
        target_height: Height to resize all videos to
        add_border: Whether to add borders between videos
        border_width: Width of borders in pixels
    """
    
    if len(video_paths) != 3:
        raise ValueError("Exactly 3 video paths are required")
    
    # Open all video captures
    caps = [cv2.VideoCapture(str(path)) for path in video_paths]
    
    # Verify all videos opened successfully
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_paths[i]}")
    
    # Get properties of all videos
    props = [get_video_properties(cap) for cap in caps]
    
    print("Video Properties:")
    for i, (path, prop) in enumerate(zip(video_paths, props)):
        print(f"  Video {i+1} ({Path(path).name}): "
              f"{prop['width']}x{prop['height']} @ {prop['fps']:.2f}fps, "
              f"{prop['frame_count']} frames")
    
    # Calculate dimensions for stitched video
    # Scale each video to target height while maintaining aspect ratio
    scaled_widths = []
    for prop in props:
        aspect_ratio = prop['width'] / prop['height']
        scaled_width = int(target_height * aspect_ratio)
        scaled_widths.append(scaled_width)
    
    # Total width includes borders
    total_border_width = border_width * 2 if add_border else 0
    total_width = sum(scaled_widths) + total_border_width
    total_height = target_height
    
    print(f"\nOutput dimensions: {total_width}x{total_height}")
    
    # Use the maximum FPS among all videos, adjusted for speed
    base_fps = max(prop['fps'] for prop in props)
    output_fps = base_fps * speed
    
    print(f"Output FPS: {output_fps:.2f} (base: {base_fps:.2f}, speed: {speed}x)")
    
    # Determine the minimum frame count (video ends when shortest video ends)
    min_frames = min(prop['frame_count'] for prop in props)
    print(f"Total frames to process: {min_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, 
                          (total_width, total_height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")
    
    # Default labels if not provided
    if labels is None:
        labels = [f"Video {i+1}" for i in range(3)]
    
    # Process frames
    frame_count = 0
    
    try:
        while True:
            frames = []
            all_read = True
            
            # Read frame from each video
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_read = False
                    break
                frames.append(frame)
            
            if not all_read:
                break
            
            # Resize frames to target dimensions
            resized_frames = []
            for i, frame in enumerate(frames):
                resized = resize_frame(frame, scaled_widths[i], target_height)
                
                # Add label
                if labels[i]:
                    resized = add_label(resized, labels[i], position='top-left',
                                       color=(0, 0, 255), font_scale=0.7)
                
                resized_frames.append(resized)
            
            # Create stitched frame
            if add_border:
                # Create border (white vertical line)
                border = np.ones((target_height, border_width, 3), dtype=np.uint8) * 255
                
                stitched = np.hstack([
                    resized_frames[0],
                    border,
                    resized_frames[1],
                    border,
                    resized_frames[2]
                ])
            else:
                stitched = np.hstack(resized_frames)
            
            # Write frame
            out.write(stitched)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / min_frames) * 100
                print(f"Progress: {frame_count}/{min_frames} frames ({progress:.1f}%)")
    
    finally:
        # Release resources
        for cap in caps:
            cap.release()
        out.release()
    
    print(f"\nComplete! Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")
    
    # Calculate output video duration
    output_duration = frame_count / output_fps
    print(f"Output video duration: {output_duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Stitch 3 videos side by side with speed control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python video_stitcher.py video1.mp4 video2.mp4 video3.mp4 -o output.mp4

  # Slow motion (0.5x speed)
  python video_stitcher.py video1.mp4 video2.mp4 video3.mp4 -o output.mp4 -s 0.5

  # With custom labels
  python video_stitcher.py video1.mp4 video2.mp4 video3.mp4 -o output.mp4 -l "Raw" "Gaussian" "Mov Avg"

  # Higher resolution output
  python video_stitcher.py video1.mp4 video2.mp4 video3.mp4 -o output.mp4 --height 720
        """
    )
    
    parser.add_argument('videos', nargs=3, help='Three input video files')
    parser.add_argument('-o', '--output', required=True, help='Output video file')
    parser.add_argument('-s', '--speed', type=float, default=1.0,
                        help='Playback speed (default: 1.0, use 0.5 for half speed)')
    parser.add_argument('-l', '--labels', nargs=3, default=None,
                        help='Labels for each video (default: Video 1, Video 2, Video 3)')
    parser.add_argument('--height', type=int, default=480,
                        help='Target height for output video (default: 480)')
    parser.add_argument('--no-border', action='store_true',
                        help='Disable borders between videos')
    parser.add_argument('--border-width', type=int, default=2,
                        help='Border width in pixels (default: 2)')
    
    args = parser.parse_args()
    
    # Validate inputs
    for video_path in args.videos:
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            return 1
    
    if args.speed <= 0:
        print("Error: Speed must be greater than 0")
        return 1
    
    # Run stitching
    try:
        stitch_videos(
            video_paths=args.videos,
            output_path=args.output,
            speed=args.speed,
            labels=args.labels,
            target_height=args.height,
            add_border=not args.no_border,
            border_width=args.border_width
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())