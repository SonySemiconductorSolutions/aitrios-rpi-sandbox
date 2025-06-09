"""
BSD 2-Clause License

Copyright (c) 2021, Raspberry Pi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import argparse
import sys
import time
import numpy as np
import cv2
import os

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# Global variables to keep track of previous and last squat state
last_boxes = None
last_scores = None
last_keypoints = None
WINDOW_SIZE_H_W = (480, 640)

prev_knee_y = None
prev_hip_y = None
last_squat_keypoints = None
last_knee_min_y = float('inf')


def calculate_angle(a, b, c):
    """Calculate angle between three points: hip - knee - ankle"""
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    if norm_ab == 0 or norm_bc == 0:
        return 0

    cosine_angle = np.dot(ab, bc) / (norm_ab * norm_bc)
    angle = 180 - np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    #print("Angle:", angle)
    return angle


squat_angle_list = []
last_feedback_text = []
feedback_timer = 0
squat_count = 0
def analyze_squat_posture(keypoints):
    """Analyze squat posture and determine whether it's correct."""
    global prev_knee_y, prev_hip_y, last_squat_keypoints, last_knee_min_y, squat_angle_list, squat_count
    feedback = []

    # Extract left leg keypoints (COCO format: 11 - hip, 13 - knee, 15 - ankle)
    hip = keypoints[11][:2]
    knee = keypoints[13][:2]
    ankle = keypoints[15][:2]

    # Confidence scores
    hip_conf = keypoints[11][2]
    knee_conf = keypoints[13][2]
    ankle_conf = keypoints[15][2]

    # Skip if confidence is too low
    if hip_conf < 0.2 or knee_conf < 0.2 or ankle_conf < 0.2:
        return []

    hip_y = hip[1]
    hip_x = hip[0]
    knee_y = knee[1]
    knee_x = knee[0]
    ankle_x = ankle[0]

    horizontal_knee_ankle_dist = abs(knee_x - ankle_x)
    horizontal_hip_knee_dist= abs(hip_x - knee_x)
    # Initialize previous frame values
    if prev_knee_y is None or prev_hip_y is None:
        prev_knee_y = knee_y
        prev_hip_y = hip_y
        return []

    # Calculate current frame angle
    current_angle = calculate_angle(hip, knee, ankle)
    #av_horizontal_knee_ankle_dist = 
    # Detect squat start or in squat phase

    if hip_y > prev_hip_y + 20 or horizontal_knee_ankle_dist > 23:
        # Still squatting ? accumulate angles
        squat_angle_list.append(current_angle)
        last_squat_keypoints = keypoints.copy()
        last_knee_min_y = knee_y

    # Detect rising from squat (angle increasing, knee rising)
    elif last_squat_keypoints is not None and knee_y < last_knee_min_y - 10:
        # Process squat angle list
        if len(squat_angle_list) >= 3:
            # Get smallest 3 angles and average
            sorted_angles = sorted(squat_angle_list)[:3]
            avg_angle = sum(sorted_angles) / len(sorted_angles)
        else:
            avg_angle = current_angle  # fallback if too few data


        # Feedback based on average squat angle
        if avg_angle > 110:
            feedback.append("Squat too shallow")
            os.system('espeak "too shallow" &')
            squat_count += 1
        elif avg_angle < 65:
            feedback.append("Squat too deep")
            os.system('espeak "too deep" &')
            squat_count += 1
        else:
            feedback.append("Good squat!")
            os.system('espeak "Good" &')
            squat_count += 1


        print("Final Squat Feedback:", feedback)
        #squat_count += 1
        # Reset state
        last_squat_keypoints = None
        last_knee_min_y = float('inf')
        squat_angle_list = []  # clear angle history for next squat

    # Update previous Y values
    prev_knee_y = knee_y
    prev_hip_y = hip_y

    return feedback



def ai_output_tensor_parse(metadata: dict):
    """Parse the output tensor into keypoints and boxes."""
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=args.detection_threshold,
            network_postprocess=True
        )
        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
    return last_boxes, last_scores, last_keypoints



def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    global feedback_timer, last_feedback_text
    """Draw detection results and feedback on screen."""
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
                        # Analyze squat and get feedback + angle
            feedback = analyze_squat_posture(keypoints[0])  # Only analyze the first person

            if feedback:
                last_feedback_text = feedback
                feedback_timer = 200  # display for 20 frames (~2 seconds if 10fps)
            # Visualize keypoints + feedback
            drawer.annotate_image(
                m.array, boxes, scores, np.zeros(scores.shape), keypoints,
                args.detection_threshold, args.detection_threshold,
                request.get_metadata(), picam2, stream
            )

            # Calculate squat angle and draw on screen (left leg)
            if keypoints[0][11][2] > 0.2 and keypoints[0][13][2] > 0.2 and keypoints[0][15][2] > 0.2:
                hip = keypoints[0][11][:2]
                knee = keypoints[0][13][:2]
                ankle = keypoints[0][15][:2]
                angle = calculate_angle(hip, knee, ankle)

                # Draw squat angle near the knee point

                angle_text = f"Angle: {int(angle)}"
                knee_point = tuple(map(int, knee))
                cv2.putText(m.array, angle_text, (knee_point[0] + 10, knee_point[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw feedback in top-left corner
            text = f"Squat Count: {squat_count}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = m.array.shape[1] - text_size[0] - 10  # Right-aligned
            text_y = 30
            cv2.putText(m.array, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            if last_feedback_text:
                for idx, line in enumerate(last_feedback_text):
                    cv2.putText(m.array, line, (10, 30 + 30 * idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def picamera2_pre_callback(request: CompletedRequest):
    """Preprocessing callback for each camera frame."""
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    ai_output_tensor_draw(request, boxes, scores, keypoints)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true")
    return parser.parse_args()


def get_drawer():
    categories = intrinsics.labels
    categories = [c for c in categories if c and c != "-"]
    return COCODrawer(categories, imx500, needs_rescale_coords=False)


if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        exit()

    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    drawer = get_drawer()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    while True:
        time.sleep(0.5)
