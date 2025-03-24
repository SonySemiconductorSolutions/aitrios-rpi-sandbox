# README



In this article, I will introduce a method for applying two different AI models on scenes captured by two cameras.


To perform different AI model using two cameras, previous versions of the Raspberry Pi (before Raspberry Pi 5) required two cameras and two Raspberry Pi boards.However, with the Raspberry Pi 5, the number of camera inputs has increased from one to two, allowing up to two cameras to be connected to a single Raspberry Pi.Therefore, in this article, I will introduce a method to connect two Raspberry Pi AI Cameras to a single Raspberry Pi 5 and run different AI models on them.

You can also check the document in [Qiita](URL) with the application.

# 1. Prerequisites

First update

    sudo apt update && sudo apt full-upgrade 

Install the imx500

    sudo apt install imx500-all

confirm the Linux Kernel version

    uname -a
    Linux raspberrypi 6.6.62+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.62-1+rpt1 (2024-11-25) aarch64 GNU/Linux

# 2. Connect Raspberry pi5 with two ai camera

Connect Raspberry pi5 with 2 imx500 ai camera, and confirm the information as follows

    $ rpicam-hello --list-cameras Available cameras
     ----------------- 
    0 : imx500 [4056x3040 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@88000/imx500@1a) 
    Modes: 'SRGGB10_CSI2P' : 2028x1520 [30.02 fps - (0, 0)/4056x3040 crop] 4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]
    
    1 : imx500 [4056x3040 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a) 
    Modes: 'SRGGB10_CSI2P' : 2028x1520 [30.02 fps - (0, 0)/4056x3040 crop] 4056x3040 [10.00 fps - (0, 0)/4056x3040 crop]

# 3. Script file introduction
| file | description |
|------|------------|
| README.md | introduction of usage |
| imx500_classification_demo_roi.py | classification ROI script |
| imx500_object_detection_demo_roi.py | object detection ROI script |
| imx500_pose_estimation_higherhrnet_demo_roi.py | pose estimation ROI script |
| imx500_segmentation_demo_roi.py | segmentation ROI script |


# Usage for each script
## 1. imx500_classification_demo_roi.py
```
usage: imx500_classification_demo_roi.py [-h] [--model MODEL] [--fps FPS]
                                         [-s | --softmax | --no-softmax]
                                         [-r | --preserve-aspect-ratio | --no-preserve-aspect-ratio]
                                         [--labels LABELS]
                                         [--print-intrinsics] [-i]
                                         [-x INPUT_TENSOR_SCALE] [-n] [-e]
                                         [-w] [--inference-roi INFERENCE_ROI]
                                         [-t] [--camera-num CAMERA_NUM]
                                         [--show-preview-roi]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path of the model
  --fps FPS             Frames per second
  -s, --softmax, --no-softmax
                        Add post-process softmax
  -r, --preserve-aspect-ratio, --no-preserve-aspect-ratio
                        preprocess the image with preserve aspect ratio
  --labels LABELS       Path to the labels file
  --print-intrinsics    Print JSON network_intrinsics then exit
  -i, --show-input-tensor
                        show input tensor
  -x INPUT_TENSOR_SCALE, --input-tensor-scale INPUT_TENSOR_SCALE
                        Display input tensor scale(x)
  -n, --no-preview      Preview off
  -e, --disable-ae      Disable Ae
  -w, --disable-awb     Disable Awb
  --inference-roi INFERENCE_ROI
                        Inference roi x,y,width,height
  -t, --show-time-measure
                        show time measure result
  --camera-num CAMERA_NUM
                        Camera number 0 or 1(default)
  --show-preview-roi    show preview of roi
```
## 2. imx500_object_detection_demo_roi.py
```
usage: imx500_object_detection_demo_roi.py [-h] [--model MODEL] [--fps FPS]
                                           [--bbox-normalization | --no-bbox-normalization]
                                           [--bbox-order {yx,xy}]
                                           [--threshold THRESHOLD] [--iou IOU]
                                           [--max-detections MAX_DETECTIONS]
                                           [--ignore-dash-labels | --no-ignore-dash-labels]
                                           [--postprocess {,nanodet}]
                                           [-r | --preserve-aspect-ratio | --no-preserve-aspect-ratio]
                                           [--labels LABELS]
                                           [--print-intrinsics] [-i]
                                           [-x INPUT_TENSOR_SCALE] [-n] [-e]
                                           [-w]
                                           [--inference-roi INFERENCE_ROI]
                                           [-t] [--camera-num CAMERA_NUM]
                                           [--show-preview-roi]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path of the model
  --fps FPS             Frames per second
  --bbox-normalization, --no-bbox-normalization
                        Normalize bbox
  --bbox-order {yx,xy}  Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0,
                        x1, y1)
  --threshold THRESHOLD
                        Detection threshold
  --iou IOU             Set iou threshold
  --max-detections MAX_DETECTIONS
                        Set max detections
  --ignore-dash-labels, --no-ignore-dash-labels
                        Remove '-' labels
  --postprocess {,nanodet}
                        Run post process of type
  -r, --preserve-aspect-ratio, --no-preserve-aspect-ratio
                        preserve the pixel aspect ratio of the input tensor
  --labels LABELS       Path to the labels file
  --print-intrinsics    Print JSON network_intrinsics then exit
  -i, --show-input-tensor
                        show input tensor
  -x INPUT_TENSOR_SCALE, --input-tensor-scale INPUT_TENSOR_SCALE
                        Display input tensor scale(x)
  -n, --no-preview      Preview off
  -e, --disable-ae      Disable Ae
  -w, --disable-awb     Disable Awb
  --inference-roi INFERENCE_ROI
                        Inference roi x,y,width,height
  -t, --show-time-measure
                        show time measure result
  --camera-num CAMERA_NUM
                        Camera number 0 or 1(default)
  --show-preview-roi    show preview of roi
```
## 3.imx500_pose_estimation_higherhrnet_demo_roi.py
```
usage: imx500_pose_estimation_higherhrnet_demo_roi.py [-h] [--model MODEL]
                                                      [--fps FPS]
                                                      [--detection-threshold DETECTION_THRESHOLD]
                                                      [--labels LABELS]
                                                      [--print-intrinsics]
                                                      [-r | --preserve-aspect-ratio | --no-preserve-aspect-ratio]
                                                      [-i]
                                                      [-x INPUT_TENSOR_SCALE]
                                                      [-n] [-e] [-w]
                                                      [--inference-roi INFERENCE_ROI]
                                                      [-t]
                                                      [--camera-num CAMERA_NUM]
                                                      [--show-preview-roi]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path of the model
  --fps FPS             Frames per second
  --detection-threshold DETECTION_THRESHOLD
                        Post-process detection threshold
  --labels LABELS       Path to the labels file
  --print-intrinsics    Print JSON network_intrinsics then exit
  -r, --preserve-aspect-ratio, --no-preserve-aspect-ratio
                        preserve the pixel aspect ratio of the input tensor
  -i, --show-input-tensor
                        show input tensor
  -x INPUT_TENSOR_SCALE, --input-tensor-scale INPUT_TENSOR_SCALE
                        Display input tensor scale(x)
  -n, --no-preview      Preview off
  -e, --disable-ae      Disable Ae
  -w, --disable-awb     Disable Awb
  --inference-roi INFERENCE_ROI
                        Inference roi x,y,width,height
  -t, --show-time-measure
                        show time measure result
  --camera-num CAMERA_NUM
                        Camera number 0 or 1(default)
  --show-preview-roi    show preview of roi
```
## 4.imx500_segmentation_demo_roi.py
```
usage: imx500_segmentation_demo_roi.py [-h] [--model MODEL] [--fps FPS]
                                       [--print-intrinsics]
                                       [-r | --preserve-aspect-ratio | --no-preserve-aspect-ratio]
                                       [-i] [-x INPUT_TENSOR_SCALE] [-n] [-e]
                                       [-w] [--inference-roi INFERENCE_ROI]
                                       [-t] [--camera-num CAMERA_NUM]
                                       [--show-preview-roi]

options:
  -h, --help            show this help message and exit
  --model MODEL         Path of the model
  --fps FPS             Frames per second
  --print-intrinsics    Print JSON network_intrinsics then exit
  -r, --preserve-aspect-ratio, --no-preserve-aspect-ratio
                        preserve the pixel aspect ratio of the input tensor
  -i, --show-input-tensor
                        show input tensor
  -x INPUT_TENSOR_SCALE, --input-tensor-scale INPUT_TENSOR_SCALE
                        Display input tensor scale(x)
  -n, --no-preview      Preview off
  -e, --disable-ae      Disable Ae
  -w, --disable-awb     Disable Awb
  --inference-roi INFERENCE_ROI
                        Inference roi x,y,width,height
  -t, --show-time-measure
                        show time measure result
  --camera-num CAMERA_NUM
                        Camera number 0 or 1(default)
  --show-preview-roi    show preview of roi
```
