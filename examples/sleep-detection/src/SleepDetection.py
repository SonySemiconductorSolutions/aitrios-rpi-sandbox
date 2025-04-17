# SPDX-FileCopyrightText: 2025 Sony Semiconductor Solutions Corporation
#
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet
import cv2

# Definition of const
KEYPOINT_NAME = [
    "nose",             #0
    "leftEye",          #1
    "rightEye",         #2
    "leftEar",          #3
    "rightEar",         #4
    "leftShoulder",     #5
    "rightShoulder",    #6
    "leftElbow",        #7
    "rightElbow",       #8
    "leftWrist",        #9
    "rightWrist",       #10
    "leftHip",          #11
    "rightHip",         #12
    "leftKnee",         #13
    "rightKnee",        #14
    "leftAnkle",        #15
    "rightAnkle"        #16
]


KEYPOINT_THRESHOLD = 5
DISTANCE_THRESHOLD = 100
KEYPOINT_SCORE_THRESHOLD = 0.5
SLEEP_THRESHOLD = 10
PAUSE_AFTER_DETECTION = 0.5  # (seconds)
TEXT = "SLEEP"


# Track motionless
def track_motionless(w, h, poses_record, motionless_count):
    keypoint_count = 0
    # Calculate the coordinate difference between the previous frame and the current frame for each KeyPoint.
    for keypoint_idx in range(len(KEYPOINT_NAME)):
        if (poses_record[0].keypoint_scores[0][keypoint_idx] >= KEYPOINT_SCORE_THRESHOLD and 
            poses_record[1].keypoint_scores[0][keypoint_idx] >= KEYPOINT_SCORE_THRESHOLD):
            x0 = int(poses_record[0].keypoints[0][2 * keypoint_idx + 1] * w)
            y0 = int(poses_record[0].keypoints[0][2 * keypoint_idx] * h)
            x1 = int(poses_record[1].keypoints[0][2 * keypoint_idx + 1] * w)
            y1 = int(poses_record[1].keypoints[0][2 * keypoint_idx] * h)
            print(f"{KEYPOINT_NAME[keypoint_idx]}   x:{x0}, y:{y0}")
            
            distance = (x1 - x0)**2 + (y1 - y0)**2
            if distance < DISTANCE_THRESHOLD:
                keypoint_count += 1
    
    if keypoint_count >= KEYPOINT_THRESHOLD:
        print("[INFO] Detected 5 or more KeyPoints.")
        motionless_count += 1
    else:
        motionless_count = 0
    
    return motionless_count

# Display Text
def display_text(image):
    print("[INFO] Display Text")
    cv2.putText(image,
            text=TEXT,
            org=(100, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2.0,
            color=(0, 0, 255),
            thickness=3) 


# Detection by modlib
def detect_sleep():
    print("[INFO] Starting sleep detection.")
    device = AiCamera()
    model = Posenet()
    device.deploy(model)

    annotator = Annotator()

    last_detection_time = 0
    motionless_count = 0
    poses_record = []

    with device as stream:
        for frame in stream:
            current_time = time.time()
            if current_time - last_detection_time > PAUSE_AFTER_DETECTION:
                last_detection_time = time.time()

                h, w, _ = frame.image.shape

                poses = frame.detections
                poses_record.append(poses)
                if len(poses_record) > 2:
                    poses_record.pop(0)
                    motionless_count = track_motionless(w, h, poses_record, motionless_count)

                print(f"Motionless Count: {motionless_count}")
                if motionless_count > SLEEP_THRESHOLD:
                    print("[INFO] Detect Sleep")
                    display_text(frame.image)

                annotator.annotate_poses(frame, frame.detections)
                frame.display()

# Main
if __name__ == "__main__":
    detect_sleep()