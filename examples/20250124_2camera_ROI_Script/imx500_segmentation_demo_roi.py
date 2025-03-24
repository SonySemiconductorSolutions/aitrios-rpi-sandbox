
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
from typing import Dict

import cv2
import numpy as np

from picamera2 import CompletedRequest, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

COLOURS = np.loadtxt("assets/colours.txt")

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

def create_and_draw_masks(request: CompletedRequest):
    """Create masks from the output tensor and draw them on the main output image."""
    masks = create_masks(request)
    draw_masks(masks)

def create_masks(request: CompletedRequest) -> Dict[int, np.ndarray]:
    """Create masks from the output tensor, scaled to the ISP output."""
    res = {}
    np_outputs = imx500.get_outputs(metadata=request.get_metadata())
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return res
    mask = np_outputs[0]
    found_indices = np.unique(mask)

    for i in found_indices:
        if i == 0:
            continue
        output_shape = [input_h, input_w, 4]
        colour = [(0, 0, 0, 0), COLOURS[int(i)]]
        colour[1][3] = 150  # update the alpha value here, to save setting it later
        overlay = np.array(mask == i, dtype=np.uint8)
        overlay = np.array(colour)[overlay].reshape(output_shape).astype(np.uint8)
        # No need to resize the overlay, it will be stretched to the output window.
        res[i] = overlay
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
    return res

def draw_masks(masks: Dict[int, np.ndarray]):
#    """Draw the masks for this request onto the ISP output."""
    if not masks:
        return
    input_w, input_h = imx500.get_input_size()
    output_shape = [input_h, input_w, 4]
    overlay = np.zeros(output_shape, dtype=np.uint8)
    if masks:
        #->added
        if args.inference_roi:
            global full_width, full_height, win_width, win_height
            b_x=int(input_w*args.inference_roi[0]/full_width)
            b_y=int(input_h*args.inference_roi[1]/full_height)
            b_w=int(input_w*args.inference_roi[2]/full_width)
            b_h=int(input_h*args.inference_roi[3]/full_height)
        if args.inference_roi and not args.show_preview_roi:
            for v in masks.values():
                roi_mask = np.zeros(output_shape, dtype=np.uint8)
                img=cv2.resize(v, (b_w, b_h))
                roi_mask[b_y:b_y+b_h, b_x:b_x+b_w]=img
                overlay += roi_mask
        else:        
        #<-added
            for v in masks.values():
                overlay += v
        #->added
        if args.inference_roi and not args.show_preview_roi:
            # Drawing ROI box
            roi_rect = np.zeros(output_shape, dtype=np.uint8)
            color = (255, 0, 0)  # red
            cv2.putText(roi_rect, "ROI", (b_x + 3, b_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(roi_rect, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
            roi_rect[:, :,3]=roi_rect[:, :,0]   #red pixel alpha set to 255
            overlay += roi_rect
        #<-added
        # Set Alphas and overlay
        picam2.set_overlay(overlay)
        #->added
        #zoom
        if args.inference_roi and args.show_preview_roi:
            picam2.set_controls({"ScalerCrop": args.inference_roi})
        #<-added

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_deeplabv3plus.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    #added->
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
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
    #added<-
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
        intrinsics.task = "segmentation"
    elif intrinsics.task != "segmentation":
        print("Network is not a segmentation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    #added->
    #config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate,\
            "CnnEnableInputTensor": True if args.show_input_tensor else False, \
            "AeEnable": False if args.disable_ae else True, \
            "AwbEnable": False if args.disable_awb else True}, buffer_count=12)
    #added<-
    imx500.show_network_fw_progress_bar()
    #added->
    #picam2.start(config, show_preview=True)
    picam2.start(config, show_preview=not args.no_preview)
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
    picam2.pre_callback = create_and_draw_masks

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
                        if (run_cnt % 10) == 0: 
                            print(run_cnt//10, "dnn_runtime:", dnn_run/10, "(ms), dsp_runtime:", dsp_run/10, "(ms)")
                            dnn_run=0.0
                            dsp_run=0.0
                except KeyError:
                    err=-1
            if args.show_input_tensor:
                try:
                    input_tensor = picam2.capture_metadata()["CnnInputTensor"]
                    if INPUT_TENSOR_SIZE != (0, 0):
                        # WINDOW
                        cv2.namedWindow("Input Tensor", cv2.WINDOW_NORMAL)
                        cv2.imshow("Input Tensor",imx500.input_tensor_image(input_tensor))
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
