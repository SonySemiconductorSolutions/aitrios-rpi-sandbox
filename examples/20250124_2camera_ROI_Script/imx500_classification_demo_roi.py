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
from typing import List

import cv2
import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

from datetime import datetime

last_detections = []
LABELS = None

#->added
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

class Classification:
    def __init__(self, idx: int, score: float):
        """Create a Classification object, recording the idx and score."""
        self.idx = idx
        self.score = score


def get_label(request: CompletedRequest, idx: int) -> str:
    """Retrieve the label corresponding to the classification index."""
    global LABELS
    if LABELS is None:
        LABELS = intrinsics.labels
        assert len(LABELS) in [1000, 1001], "Labels file should contain 1000 or 1001 labels."
        output_tensor_size = imx500.get_output_shapes(request.get_metadata())[0][0]
        if output_tensor_size == 1000:
            LABELS = LABELS[1:]  # Ignore the background label if present
    return LABELS[idx]


def parse_and_draw_classification_results(request: CompletedRequest):
    """Analyse and draw the classification results in the output tensor."""
    results = parse_classification_results(request)
    draw_classification_results(request, results)

def parse_classification_results(request: CompletedRequest) -> List[Classification]:
    """Parse the output tensor into the classification results above the threshold."""
    global last_detections
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        return last_detections
    np_output = np_outputs[0]
    if intrinsics.softmax:
        np_output = softmax(np_output)
    top_indices = np.argpartition(-np_output, 3)[:3]  # Get top 3 indices with the highest scores
    top_indices = top_indices[np.argsort(-np_output[top_indices])]  # Sort the top 3 indices by their scores
    last_detections = [Classification(index, np_output[index]) for index in top_indices]
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
            #print(cnt,":",fps,":",delta,":",t0)
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


def draw_classification_results(request: CompletedRequest, results: List[Classification], stream: str = "main"):
    """Draw the classification results for this request onto the ISP output."""
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
        #if intrinsics.preserve_aspect_ratio:
        if (intrinsics.preserve_aspect_ratio or args.inference_roi) and not args.show_preview_roi:
        #<-added
            # Drawing ROI box
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
            text_left, text_top = b_x, b_y + 20
            #->added
            if args.inference_roi:
                text_left, text_top = 0, 0
            #<-added
        else:
            text_left, text_top = 0, 0
        # Drawing labels (in the ROI box if it exists)
        for index, result in enumerate(results):
            label = get_label(request, idx=result.idx)
            text = f"{label}: {result.score:.3f}"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = text_left + 5
            text_y = text_top + 15 + index * 20

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.3
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("-s", "--softmax", action=argparse.BooleanOptionalAction, help="Add post-process softmax")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preprocess the image with preserve aspect ratio")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    #added->
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
        intrinsics.task = "classification"
    elif intrinsics.task != "classification":
        print("Network is not a classification task", file=sys.stderr)
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
        with open("assets/imagenet_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    #added->
    #config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate,\
            "CnnEnableInputTensor": True if args.show_input_tensor else False, \
            "AeEnable": False if args.disable_ae else True, \
            "AwbEnable": False if args.disable_awb else True}, buffer_count=12)
    #<-added

    imx500.show_network_fw_progress_bar()
    #added->
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

    # Register the callback to parse and draw classification results
    picam2.pre_callback = parse_and_draw_classification_results

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
    if args.model=="/usr/share/imx500-models/imx500_network_mobilevit_xs.rpk":
        input_tensor_rgb = True
    #added<-
    #added->
    try:    
        while True:
            #added->
            #time.sleep(0.5)
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
                            #print(run_cnt//100, "dnn_runtime:", dnn_run/100, "(ms), dsp_runtime:", dsp_run/100, "(ms)")
                            print(run_cnt//10, "dnn_runtime:", dnn_run/10, "(ms), dsp_runtime:", dsp_run/10, "(ms)")
                            #print("dnn_runtime:", kpi_info[0]/1000, "(ms), dsp_runtime:", kpi_info[1]/1000, "(ms)")
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
                        #WINDOW
                        cv2.namedWindow("Input Tensor", cv2.WINDOW_NORMAL)
                        img=imx500.input_tensor_image(input_tensor)
                        #convert color
                        if input_tensor_rgb:
                            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cv2.imshow("Input Tensor",img)
                        #cv2.imshow("Input Tensor",imx500.input_tensor_image(input_tensor))
                        cv2.resizeWindow("Input Tensor", *INPUT_TENSOR_SIZE)
                        #must be wait
                        cv2.waitKey(1)
                except KeyError:
                    pass
            else:
                time.sleep(0.5)
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