from __future__ import annotations

import json
from tqdm import tqdm
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from accelerate import Accelerator

from argparse import ArgumentParser
from pathlib import Path
import cv2

from utils.shared_utils import get_video_rotation, auto_oriented_frame
        
def process_frame(person_preprocessor: AutoProcessor,
                  person_model: RTDetrForObjectDetection, 
                  pose_preprocessor: AutoProcessor,
                  pose_model: VitPoseForPoseEstimation,
                  input_image: Image.Image, 
                  annotate: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Image.Image | None]:
    '''
    Note: just cropping the image to the field bounds before processing does not speed up inference.
    '''
    inputs = person_preprocessor(images=input_image, return_tensors="pt").to(person_model.device)
    with torch.no_grad():
        detections = person_model(**inputs)
        
    pp_detections = person_preprocessor.post_process_object_detection(
        detections, 
        target_sizes=torch.tensor([(input_image.height, input_image.width)]), 
        threshold=0.3
    )
    
    pp_detections = pp_detections[0]

    # Human label refers 0 index in COCO dataset
    person_boxes = pp_detections["boxes"][pp_detections["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    
    if len(person_boxes) == 0: # No persons detected
        empty_array = np.array([])
        if not annotate:
            return empty_array, empty_array, None
        else:
            return empty_array, empty_array, input_image

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    inputs = pose_preprocessor(input_image, boxes=[person_boxes], return_tensors="pt").to(pose_model.device)

    with torch.no_grad():
        outputs = pose_model(**inputs)

    pose_results = pose_preprocessor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]

    xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
    scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

    if not annotate:
        return xy, scores, None
    
    key_points = sv.KeyPoints(
        xy=xy, confidence=scores
    )

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=1
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.RED,
        radius=2
    )
    ankle_annotator = sv.VertexAnnotator(
        color=sv.Color.YELLOW,
        radius=2
    )

    annotated_frame = edge_annotator.annotate(
        scene=input_image.copy(),
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    
    annotated_frame = ankle_annotator.annotate(
        scene=annotated_frame,
        key_points=sv.KeyPoints(xy=xy[:, 15:17], confidence=scores[:, 15:17])

    )
    
    return xy, scores, annotated_frame

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_video', type=Path, required=True, help='Path to original video file')
    parser.add_argument('--stab_video', type=Path, required=True, help='Path to stabilized video file')
    parser.add_argument('--output', type=Path, required=False, help='Path to output image or video file')
    parser.add_argument('--person_detector', default='PekingU/rtdetr_r50vd_coco_o365', help='Person detector model name')
    parser.add_argument('--keypoint_detector', default='usyd-community/vitpose-base-simple', help='Keypoint detection model name')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    device = Accelerator().device
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Detect humans in the image
    print("Loading person detection model...")
    person_image_processor = AutoProcessor.from_pretrained(args.person_detector, use_fast=True)
    person_model = RTDetrForObjectDetection.from_pretrained(args.person_detector, device_map=device)
    
    # Detect keypoints for each person found
    print("Loading keypoint detection model...")
    image_processor = AutoProcessor.from_pretrained(args.keypoint_detector, use_fast=True)
    pose_model = VitPoseForPoseEstimation.from_pretrained(args.keypoint_detector, device_map=device)
    video_pose_estimations = {}
    trajectory_annotator = sv.TraceAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    if args.debug:
        print("Annotating frames")
        
    out_file = args.output / f'{args.stab_video.stem}_pose_estimation.mp4'
    cap = cv2.VideoCapture(str(args.stab_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotation = get_video_rotation(args.input_video)

    if out_file.exists() and args.debug:
        out_file.unlink()
    
    if  args.debug:
        out = cv2.VideoWriter(out_file, 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            fps,
                            (width, height))
    if fps > 60:
        fps_step_size = int(fps // 60)
    else:
        fps_step_size = 1

    for frame_idx in tqdm(range(0, total_frames, fps_step_size)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = auto_oriented_frame(frame, rotation)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        xy, scores, annotated_frame = process_frame(person_preprocessor=person_image_processor,
                                                                    person_model=person_model,
                                                                    pose_preprocessor=image_processor,
                                                                    pose_model=pose_model,
                                                                    input_image=pil_image,
                                                                    annotate=args.debug,
                                                                )
        
        if len(xy) == 0:
            pass # no keypoints detected in this frame
        else:
            video_pose_estimations[frame_idx] = {'positions': xy.tolist(), 
                                                 'scores': scores.tolist()}
            
            
            if args.debug:
                annotated_frame_bgr = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
                out.write(annotated_frame_bgr)
    cap.release()
    
    if args.debug:
        out.release()
    
    pose_est_file = args.output / 'pose_estimations.json'
    with pose_est_file.open('w') as f:
        json.dump(video_pose_estimations, f)
