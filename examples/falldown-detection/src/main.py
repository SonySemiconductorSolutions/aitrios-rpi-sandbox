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
import cv2
import time
import math

import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

last_boxes = None
last_scores = None
last_keypoints = None
WINDOW_SIZE_H_W = (480, 640)


class FalldownDrawer(COCODrawer):
    def __init__(self, categories, imx500, needs_rescale_coords=True):
        super().__init__(categories, imx500, needs_rescale_coords)
        self.fall_label = ['Fall', 'Normal', 'Unknown']
        self.fall_color = [(0,0,255), (0,255,0), [128,128,128]]
    
    def draw_bounding_box(self, img, annotation, class_id, score, metadata: dict, picam2: Picamera2, stream):
        y0, x0, y1, x1 = self.get_coords(annotation, metadata, picam2, stream)
        text = f"{self.fall_label[int(class_id)]}:{score:.3f}"
        cv2.rectangle(img, (x0, y0), (x1, y1), self.fall_color[int(class_id)], 2)
        cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_fall_keypoints(self, img, keypoints, falldown_threshold, metadata: dict, picam2: Picamera2, stream):

        def scale_point(xy):
            x0 = xy[0]
            y0 = xy[1]
            y0, x0, _, _ = self.get_coords((y0, x0, y0 + 1, x0 + 1), metadata, picam2, stream)
            return x0, y0
        
        T1 = (100 - falldown_threshold, 320)
        T2 = (100 + falldown_threshold, 320)

        tan_threshold = abs(T1[1] - 400) / abs(T1[0] - 100) + 0.000000001
        
        fall_ave_points = [
            [0, 1, 2, 3, 4], # Head
            [13, 14], # Knees
            [15, 16] # Feet
        ]
        fall_keypoints = []
        for fall_ave in fall_ave_points:
            kp_filter = [k for k in keypoints[fall_ave] if k[2] > 0]
            if len(kp_filter) < 1:
                return 2, 0
            else:
                fall_keypoints.append(np.average(np.array(kp_filter), axis=-2))
        head = scale_point(fall_keypoints[0])
        knees = scale_point(fall_keypoints[1])
        feet = scale_point(fall_keypoints[2])
        (dx1, dy1) = (head[0] - feet[0], head[1] - feet[1])

        # if fall -> color change
        tan_angle = abs(dy1) / (abs(dx1) + 0.000000001)
        cls_id = 0

        if (tan_angle > tan_threshold):
            # draw selektons
            cv2.line(img, head, knees, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.line(img, knees, feet, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cls_id = 1
        else:
            cv2.line(img, head, knees, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(img,knees, feet, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # draw points
        cv2.circle(img, head, 3, (0, 0, 255), 5, lineType=cv2.LINE_AA)
        cv2.circle(img, knees, 3, (0, 255, 0), 5, lineType=cv2.LINE_AA)
        cv2.circle(img, feet, 3, (255, 0, 0), 5, lineType=cv2.LINE_AA)

        return cls_id, math.atan(tan_angle) / math.pi * 180


        
    
    def annotate_image(self, img, b, s, c, k, box_min_conf, fall_thres, metadata, picam2, stream):
        for index, row in enumerate(b):
            if s[index] >= box_min_conf:
                clsid, thres = 1, None
                if k is not None:
                    clsid, thres = self.draw_fall_keypoints(img, k[index], fall_thres, metadata, picam2, stream)
                self.draw_bounding_box(img, row, clsid, thres, metadata, picam2, stream)



def ai_output_tensor_parse(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(outputs=np_outputs,
                                                           img_size=WINDOW_SIZE_H_W,
                                                           img_w_pad=(0, 0),
                                                           img_h_pad=(0, 0),
                                                           detection_threshold=args.detection_threshold,
                                                           network_postprocess=True)

        if scores is not None and len(scores) > 0:
            last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
            last_boxes = [np.array(b) for b in boxes]
            last_scores = np.array(scores)
    return last_boxes, last_scores, last_keypoints


def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
    """Draw the detections for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        if boxes is not None and len(boxes) > 0:
            drawer.annotate_image(m.array, boxes, scores,
                                  np.zeros(scores.shape), keypoints, args.detection_threshold,
                                  args.fall_threshold, request.get_metadata(), picam2, stream)


def picamera2_pre_callback(request: CompletedRequest):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
    ai_output_tensor_draw(request, boxes, scores, keypoints)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model", default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--detection-threshold", type=float, default=0.3, help="Post-process detection threshold")
    parser.add_argument("--fall-threshold", type=float, default=45, help='Falldown angle threshold')
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


def get_drawer():
    categories = intrinsics.labels
    categories = [c for c in categories if c and c != "-"]
    return FalldownDrawer(categories, imx500, needs_rescale_coords=False)


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("Network is not a pose estimation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.inference_rate is None:
        intrinsics.inference_rate = 10
    if intrinsics.labels is None:
        with open("coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    drawer = get_drawer()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    while True:
        time.sleep(0.5)
