# BSD 2-Clause License
# 
# Copyright (c) 2021, Raspberry Pi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
from functools import lru_cache
from collections import deque
import os
import io
import time

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# -------------------------------
# Flask App Setup and Endpoints
# -------------------------------

app = Flask(__name__)

# Fixed-length history (30 frames) for detection counts and categories (initialized with default values)
detection_count_history = deque([0] * 30, maxlen=30)
categories_history = deque([[] for _ in range(30)], maxlen=30)
last_update_time = 0.0  # Update timing (update once per second)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection_data')
def detection_data():
    """
    Calculate and return the detection counts per class based on the detection results for each frame.
    Example output:
    {
      "frameLabels": [1, 2, 3, ..., 30],
      "datasets": [
         {
           "label": "person",
           "data": [0, 1, 0, ...],
           "borderColor": "rgba(255, 99, 132, 1)",
           "fill": false,
           "tension": 0.1
         },
         { ... }
      ]
    }
    """
    # Use the fixed-length history (already initialized)
    history = list(categories_history)
    # Frame labels are always 1 to 30
    frame_indices = list(range(1, 31))
    global_classes = set()
    # Gather all classes present in the history
    for frame in history:
        global_classes.update(frame)
    # Create a list of detection counts for each class over the 30 frames
    per_class_counts = {c: [] for c in global_classes}
    for frame in history:
        counter = {}
        for c in frame:
            counter[c] = counter.get(c, 0) + 1
        for c in global_classes:
            per_class_counts[c].append(counter.get(c, 0))
    
    # Retrieve label information (if not available, use the numeric string)
    labels_list = get_labels()
    datasets = []
    # Predefined colors (will cycle if there are more classes than colors)
    colors = [
        "rgba(255, 99, 132, 1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(255, 159, 64, 1)"
    ]
    i = 0
    for c, counts in per_class_counts.items():
        # Use the label from labels_list if available, otherwise use the numeric value as a string
        class_label = labels_list[int(c)] if int(c) < len(labels_list) else str(int(c))
        datasets.append({
            "label": class_label,
            "data": counts,
            "borderColor": colors[i % len(colors)],
            "fill": False,
            "tension": 0.1
        })
        i += 1

    return jsonify(frameLabels=frame_indices, datasets=datasets)

@app.route('/video_feed')
def video_feed():
    """
    Multipart JPEG streaming endpoint.
    The frames obtained from Picamera2's capture_array() have bounding boxes drawn via the pre_callback.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# Global Variables
# -------------------------------
last_detections = []  # Latest detection results
last_results = None   # Variable used by the draw_detections callback
args = None
imx500 = None
intrinsics = None
picam2 = None

# -------------------------------
# Command-Line Argument Parsing
# -------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction,
                        help="Normalize bounding box values")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order: yx -> (y0, x0, y1, x1), xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set IOU threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set maximum number of detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction,
                        help="Remove '-' labels")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post-process of specified type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="Preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

# -------------------------------
# Detection Result Class
# -------------------------------
class Detection:
    def __init__(self, coords, category, conf, metadata):
        """
        coords: Coordinates obtained from inference (before transformation)
        category: Class ID (int)
        conf: Confidence score (float)
        metadata: Dictionary obtained from capture_metadata()
        """
        self.category = category
        self.conf = conf
        # Convert the coordinates to ISP output coordinates [x, y, w, h] using IMX500's conversion function
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

# -------------------------------
# Parse Detection Results
# -------------------------------
def parse_detections(metadata: dict):
    """
    Parse the output tensor to create a list of Detection objects.
    Coordinate conversion for visualization is handled in Detection.__init__.
    """
    global last_detections, last_results

    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()

    if np_outputs is None:
        return last_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=threshold,
            iou_thres=iou,
            max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        # Split each element into [x, y, w, h]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = list(zip(*boxes))

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    last_results = last_detections
    update_detection_data(last_detections)
    return last_detections

# -------------------------------
# Get Labels (Cached)
# -------------------------------
@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

# -------------------------------
# Draw Bounding Boxes and Labels
# -------------------------------
def draw_detections(request, stream="main"):
    """
    Draw bounding boxes and labels on the ISP output image.
    """
    global last_results
    detections = last_results
    if detections is None:
        return

    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box

            if labels and int(detection.category) < len(labels):
                label_text = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            else:
                label_text = f"{detection.category} ({detection.conf:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            text_x = x + 5
            text_y = y + 15

            # Draw a semi-transparent background for the text
            overlay = m.array.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),
                cv2.FILLED
            )
            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw the text
            cv2.putText(
                m.array, label_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

            # Draw the detection bounding box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # If aspect ratio is preserved, display the ROI if available
        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)
            cv2.putText(
                m.array, "ROI", (b_x + 5, b_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), color, thickness=1)

# -------------------------------
# Update Detection History (Fixed 30-frame history, update once per second)
# -------------------------------
def update_detection_data(detections):
    global detection_count_history, categories_history, last_update_time
    current_time = time.time()
    if current_time - last_update_time >= 1:
        detection_count_history.append(len(detections))
        categories = [d.category for d in detections]
        categories_history.append(categories)
        last_update_time = current_time

# -------------------------------
# Generate Frames for Streaming
# -------------------------------
def generate_frames():
    while True:
        metadata = picam2.capture_metadata()
        parse_detections(metadata)
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()
    if not hasattr(intrinsics, 'bbox_order'):
        intrinsics.bbox_order = "yx"

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections

    app.run(host='0.0.0.0', port=5000, debug=False)
