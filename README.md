# aitrios-rpi-sandbox-dev

## Sample Applications 💻

<div align="center">
<p align="center">

  Sample&nbsp;Application  | Description&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; | AI&nbsp;Model&nbsp;Type | Model&nbsp;Used 
-------------------- | -----------|--------------------|---------
[2camera ROI Script](./examples/20250124_2camera_ROI_Script) | Raspberry Pi connect to 2 camera with ROI function to run two different model at the same time| Object detection　/ Segmentaion | [ssd mobilenetv2 fpnlite 320x320 pp](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk) [DeepLabv3Plus](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_deeplabv3plus.rpk) 
[docker_pose_estimation](./examples/docker_pose_estimation) | This example contains a Dockerized Flask application that streams video from a Raspberry Pi AI Camera and performs real-time pose estimation, ensuring user privacy by displaying only skeletal information on a neutral background. | Pose Estimation | [imx500_network_higherhrnet_coco.rpk](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk) 
[docker webui](./examples/docker_webui) | This example contains a Dockerized Flask application that streams video from a Raspberry Pi AI Camera and performs real‑time object detection using Raspberry Pi AI Camera | Object detection | [imx500 network ssd mobilenetv2 fpnlite 320x320_pp](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk) 
[driver_detect](./examples/driver_detect) | Detect the cell phone for driver during driving| Object detection | [nanodet_plus_416x416_pp](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416_pp.rpk)
[evaluation tool](./examples/evaluation/README.md) | Model evaluation tool for classification, object detection, segmentation, and pose estimation (e.g. COCO mAP, classification accuracy). Runs against the AI camera (data injection) or an interpreter service (keras or onnx). | Classification, Object Detection, Semantic Segmentation, and Pose Estimation | Support for all [zoo models](https://github.com/SonySemiconductorSolutions/aitrios-rpi-model-zoo) and can be extended to custom models, see [tutorials](https://github.com/SonySemiconductorSolutions/aitrios-rpi-tutorials-ai-model-training) and [training-samples](https://github.com/SonySemiconductorSolutions/aitrios-rpi-training-samples)
[Falldown detection](./examples/falldown-detection) | Detect person is falling or not | Pose Estimation | [imx500_network_higherhrnet_coco.rpk](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk)
[gym_pose_detect](./examples/gym_pose_detect) | Using pose estimation models HigherHRNet to detect and evaluate squat posture in real time | Pose Estimation | [HigherHRNet](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk) 
[homecamera app](./examples/homecamera-app) | Detects people, triggers an alarm sound, and sends a screenshot to a messaging app. | Object Detection | [SSDMobileNetV2FPNLite320x320](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk) 
[outofstock-detection](./examples/outofstock-detection) | This system uses object detection for refrigerator inventory management, providing regular email updates with inventory information. | Object Detection | [nanodet plus 416x416](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416.rpk)  
[plc human detection](./examples/plc_human_detection) | Detectes people and nortificate it using LED lights controlled by PLC(OpenPLC)| Object Detection | [nanodet plus 416x416](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_nanodet_plus_416x416.rpk) 
[privacy mask](./examples/privacy-mask) | Detect people and apply privacy masking to the detected areas | Object Detection | [SSDMobileNetV2FPNLite320x320](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk)
[record_detection_result](./examples/record_detection_result) | Saves camera images and object detection annotations (JSON) together, with a visualization tool to overlay detection results onto saved images | Object Detection | [ssd mobilenetv2 fpnlite 320x320 pp](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk)
[sleep detection ](./examples/sleep-detection) | Detect person's keypoints, and if there is little movement for a certain number of frames, classify it as drowsiness. | Pose Estimation | [Posenet](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_posenet.rpk)
[unwell-person-detection](./examples/unwell-person-detection) | Detects potential emergencies by analyzing the movement of a person's poses within a designated area, alerting when minimal motion is detected. | Pose Estimation | [Posenet](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_posenet.rpk)
[virtual-fitting](./examples/virtual-fitting) | Estimate human pose and overlay T-shirt image on their upper body | Pose Estimation | [HigherHRNet](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk) 

</p>    
</div>

## Trademark

- [Read This First](https://developer.aitrios.sony-semicon.com/en/documents/read-this-first)

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.
