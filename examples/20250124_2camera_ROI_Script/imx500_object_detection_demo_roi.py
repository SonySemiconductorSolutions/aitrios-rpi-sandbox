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
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

last_detections = []

#->added
import time
import subprocess
cnt0=0
cnt=0
t0=0
delta_100=0
delta_min=1000
delta_max=0
cnt_max=500
full_width=4056
full_height=3040
win_width=640
win_height=480
#<-added

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    #->added
    global full_width, full_height
    #full_width, full_height = imx500.get_full_sensor_resolution()
    #added<-
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    #print("input : ", input_w, input_h)
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                          max_out_dets=max_detections)[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        #->added
        if args.inference_roi:
            boxes[...,0]=(boxes[...,0]*args.inference_roi[3]+args.inference_roi[1])/full_height
            boxes[...,1]=(boxes[...,1]*args.inference_roi[2]+args.inference_roi[0])/full_width
            boxes[...,2]=(boxes[...,2]*args.inference_roi[3]+args.inference_roi[1])/full_height
            boxes[...,3]=(boxes[...,3]*args.inference_roi[2]+args.inference_roi[0])/full_width
        #added<-
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h            
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        #->added
        if args.inference_roi:
            boxes[0]=(boxes[0]*args.inference_roi[3]+args.inference_roi[1])/full_height
            boxes[1]=(boxes[1]*args.inference_roi[2]+args.inference_roi[0])/full_width
            boxes[2]=(boxes[2]*args.inference_roi[3]+args.inference_roi[1])/full_height
            boxes[3]=(boxes[3]*args.inference_roi[2]+args.inference_roi[0])/full_width
        #added<-
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    #->added
    if args.show_time_measure:
        global cnt0, cnt, t0, delta_100, delta_min, delta_max
        t1=time.time()
        if t0 > 0:
            delta=t1-t0
        else:
            delta=0
        t0=t1 
        delta_100+=delta
        cnt+=1
        if delta != 0:
            fps=1.0/delta
            if delta>delta_max:
                delta_max=delta
            if delta<delta_min:
                delta_min=delta
        if (cnt-cnt0) == 100:
            if delta_100 != 0:
                fps=1.0/(delta_100/100.0)
                print(cnt,":",f'{fps:.3f}',"fps,",f'{delta_100*10:.3f}',"ms,",f'{delta_min*1000:.3f}',"ms<min>,",f'{delta_max*1000:.3f}',"ms<max>")
            delta_100=0
            cnt0=cnt
            delta_min=1000
            delta_max=0
    #<-added
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        #->added
        if args.inference_roi and args.show_preview_roi:
            #roi scale
            global full_width, full_height, win_width, win_height
            c_x = int(args.inference_roi[0]/full_width*win_width)
            c_y = int(args.inference_roi[1]/full_height*win_height)
            c_w = int(args.inference_roi[2]/full_width*win_width)
            c_h = int(args.inference_roi[3]/full_height*win_height)
            #roi zoom
            img0=m.array.copy()
            img1 = np.zeros([c_h,c_w,4], dtype=np.uint8)
            for i in range(4):
                img1[:,:,i]=img0[c_y:c_y+c_h,c_x:c_x+c_w,i]
            cv2.resize(img1, (win_width, win_height), m.array)
        #<-added
        for detection in detections:
            x, y, w, h = detection.box
            #->added
            if args.inference_roi and args.show_preview_roi:
                x=int((x-c_x)*full_width/args.inference_roi[2])
                y=int((y-c_y)*full_height/args.inference_roi[3])
                w=int(w*full_width/args.inference_roi[2])
                h=int(h*full_height/args.inference_roi[3])
            else:
                x, y, w, h = detection.box
            #<-added
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)                
            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

        #->added
        #if intrinsics.preserve_aspect_ratio:
        if (intrinsics.preserve_aspect_ratio or args.inference_roi) and not args.show_preview_roi:
        #<-added
            # Drawing ROI box
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    #->added
    parser.add_argument("-i","--show-input-tensor", action="store_true", help="show input tensor")
    parser.add_argument("-x", "--input-tensor-scale", type=int, default=1, help="Display input tensor scale(x)")
    parser.add_argument("-n", "--no-preview", action="store_true", help="Preview off")
    parser.add_argument("-e", "--disable-ae", action="store_true", help="Disable Ae")
    parser.add_argument("-w", "--disable-awb", action="store_true", help="Disable Awb")
    tp=lambda x:tuple(map(int, x.split(',')))
    parser.add_argument("--inference-roi", type=tp, help="Inference roi x,y,width,height")
    parser.add_argument("-t","--show-time-measure", action="store_true", help="show time measure result")
    parser.add_argument("--camera-num", type=int, help="Camera number 0 or 1(default)", default=1)
    parser.add_argument("--show-preview-roi", action="store_true", help="show preview of roi")
    #<-added
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    #->added
    #imx500 = IMX500(args.model)
    if args.camera_num == 1:
        imx500 = IMX500(args.model, 'i2c@80000')    #cam1 *default
    else:
        imx500 = IMX500(args.model, 'i2c@88000')    #cam0
    #<-added
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    #->added
    #config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate,\
            "CnnEnableInputTensor": True if args.show_input_tensor else False, \
            "AeEnable": False if args.disable_ae else True, \
            "AwbEnable": False if args.disable_awb else True}, buffer_count=12)
    #<-added

    imx500.show_network_fw_progress_bar()
    #->added
    #picam2.start(config, show_preview=True)
    picam2.start(config, show_preview=not args.no_preview)
    #<-added

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    #added->
    #set ROI
    if args.inference_roi:
        if args.inference_roi[0]+args.inference_roi[2] > full_width or \
            args.inference_roi[1]+args.inference_roi[3] > full_height:
            print("[Error] Ilegal roi size !")
            exit()
        imx500.set_inference_roi_abs(args.inference_roi)
        print("set roi")
    else:
        args.show_preview_roi=False
    #added<-

    last_results = None
    picam2.pre_callback = draw_detections
    #added->
    #show input tensor prepare
    INPUT_TENSOR_SIZE = [0, 0]
    if args.show_input_tensor:
        err=0
        for _ in range(10):
            try:
                t = picam2.capture_metadata()["CnnInputTensorInfo"]
                network_name, width, height, num_channels = imx500._IMX500__get_input_tensor_info(t)
                break
            except KeyError:
                err=-1
        if err == 0:
            INPUT_TENSOR_SIZE = [height*args.input_tensor_scale, width*args.input_tensor_scale]
            cv2.startWindowThread()
    #added<-

    #added->
    dnn_run=0.0
    dsp_run=0.0
    run_cnt=0
    input_tensor_rgb = False
    if args.model=="/usr/share/imx500-models/imx500_network_nanodet_plus_416x416.rpk":
        input_tensor_rgb = True
    if args.model=="/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk":
        input_tensor_rgb = True
    key=0
    #added<-
    #added->
    try:    
    #added<-
        while True:
            last_results = parse_detections(picam2.capture_metadata())
            #added->
            if args.show_time_measure:
                err=0
                try:
                    kpi_info = picam2.capture_metadata()["CnnKpiInfo"]
                    if kpi_info:
                        dnn_run=dnn_run+kpi_info[0]/1000
                        dsp_run=dsp_run+kpi_info[1]/1000
                        run_cnt=run_cnt+1
                        #if (run_cnt % 100) == 0: 
                        if (run_cnt % 10) == 0: 
                            print(run_cnt//10, "dnn_runtime:", dnn_run/10, "(ms), dsp_runtime:", dsp_run/10, "(ms)")
                            dnn_run=0.0
                            dsp_run=0.0
                except KeyError:
                    err=-1
                #if err < 0:
                #    print("no kpi info")
            if args.show_input_tensor:
                try:
                    input_tensor = picam2.capture_metadata()["CnnInputTensor"]
                    if INPUT_TENSOR_SIZE != (0, 0):
                        # WINDOW
                        cv2.namedWindow("Input Tensor", cv2.WINDOW_NORMAL)
                        img=imx500.input_tensor_image(input_tensor)
                        #convert color
                        if input_tensor_rgb:
                            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cv2.imshow("Input Tensor",img)
                        cv2.resizeWindow("Input Tensor", *INPUT_TENSOR_SIZE)
                        #must be wait
                        key=cv2.waitKey(1)
                except KeyError:
                    pass
            if args.show_time_measure:
                if cnt > cnt_max:
                    subprocess.call('date')
                    subprocess.call('grim')
                    break
            #<-added
    #added->
    except KeyboardInterrupt:
        pass
    #stop streaming and exit window
    picam2.stop()
    if args.show_input_tensor:
        cv2.destroyAllWindows()
    #<-added
