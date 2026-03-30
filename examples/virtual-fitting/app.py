#
# Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet
from modlib.apps.tracker.byte_tracker import BYTETracker

# Constants defenition
OVERLAY_IMG_X_OFFSET_RATIO = 2.0 # Decrease this to increase the leftward offset of the overlay image
OVERLAY_IMG_Y_OFFSET_RATIO = 3.5 # Decrease this to increase the upward offset of the overlay image
OVERLAY_IMG_HEIGHT_RATIO   = 2.2 # Decrease this to reduce the overlay image height
OVERLAY_IMG_WIDTH_RATIO    = 1.5 # Decrease this to reduce the overlay image width
CONFIDENCE_THRETHORLD      = 0.3 # Detection confidence threshold

class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def select_valid_coordinate(coord_a, coord_b):
    """
    Select a valid coordinate, prioritizing the first non-zero value
    """
    if coord_a == 0 and coord_b == 0:
        return 0  
    elif coord_a == 0:
        return coord_b
    else:
        return coord_a

def overlay_image_on_upper_body(frame, keypoints, overlay_image):
    """
    Overlay an image on the upper body
    """
    scene = frame.image
    height, width = scene.shape[:2]
    # Convert normalized keypoint coordinates to actual image coordinates
    left_shoulder_x  = int(keypoints[5][0] * width)
    left_shoulder_y  = int(keypoints[5][1] * height)
    right_shoulder_x = int(keypoints[6][0] * width)
    right_shoulder_y = int(keypoints[6][1] * height)
    left_hip_x       = int(keypoints[11][0] * width)
    left_hip_y       = int(keypoints[11][1] * height)
    right_hip_x      = int(keypoints[12][0] * width)
    right_hip_y      = int(keypoints[12][1] * height)
    
    # Determine upper body bounding box coordinates
    right_top_x   = select_valid_coordinate(right_shoulder_x, right_hip_x)
    right_top_y   = select_valid_coordinate(right_shoulder_y, left_shoulder_y)
    left_bottom_x = select_valid_coordinate(left_shoulder_x,  left_hip_x)
    left_bottom_y = select_valid_coordinate(right_hip_y,      left_hip_y)

    # Adjust overlay image position based on reference points
    overlay_img_x = int(right_top_x - (left_bottom_x - right_top_x) // OVERLAY_IMG_X_OFFSET_RATIO)
    overlay_img_y = int(right_top_y - (left_bottom_y - right_top_y) // OVERLAY_IMG_Y_OFFSET_RATIO)

    # Resize overlay image based on reference points
    overlay_img_h = int((left_bottom_y - right_top_y) * OVERLAY_IMG_HEIGHT_RATIO)
    overlay_img_w = int((left_bottom_x - right_top_x) * OVERLAY_IMG_WIDTH_RATIO)
    
    try:
        resized_overlay = cv2.resize(overlay_image, (overlay_img_w, overlay_img_h))
        y1 = max(0, overlay_img_y)
        y2 = min(height, overlay_img_y + overlay_img_h)
        x1 = max(0, overlay_img_x)
        x2 = min(width, overlay_img_x + overlay_img_w)
        
        oy1 = y1 - overlay_img_y
        oy2 = oy1 + (y2 - y1)
        ox1 = x1 - overlay_img_x
        ox2 = ox1 + (x2 - x1)
        
        #check if image has alpha channel 
        if resized_overlay.shape[2] == 4:
            rgb_channels = resized_overlay[:, :, :3]
            alpha_channel = resized_overlay[:, :, :3] /255.0
            alpha_channel =alpha_channel[:, :, np.newaxis]
            
            alpha_crop = alpha_channel[oy1:oy2, ox1:ox2]
            rgb_crop =rgb_channels[oy1:oy2, ox1:ox2]
            
            scene[y1:y2, x1:x2] = (alpha_crop * rgb_crop + (1 - alpha_crop) * scene[y1:y2, x1:x2]).astype(np.uint8)
        else:
            scene[y1:y2, x1:x2] = resized_overlay[oy1:oy2, ox1:ox2]
    except:
        pass
    return scene

def start_workout_demo():
    device = AiCamera()
    model = Posenet()
    device.deploy(model)

    # Load the overlay image
    overlay_image = cv2.imread("tshirt.png", cv2.IMREAD_UNCHANGED)

    tracker = BYTETracker(BYTETrackerArgs())

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > CONFIDENCE_THRETHORLD]
            detections = tracker.update(frame, detections)

            for k, _, _, _, t in detections:
                frame.image = overlay_image_on_upper_body(frame, k, overlay_image)

            frame.display()

if __name__ == "__main__":
    start_workout_demo()
    exit()
