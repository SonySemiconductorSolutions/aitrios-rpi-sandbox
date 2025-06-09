
## AI based fitness form guidance system

This Python script is designed to run on the Raspberry Pi 5 with the Sony IMX500 AI Camera, using pose estimation models HigherHRNet to detect and evaluate squat posture in real time. By analyzing key joint positions from the camera, it determines whether a user's squat is too shallow, good, or too deep, and automatically counts valid squats.

The script currently analyzes only the left side of the body. You can modify it to evaluate the full body or the right side by adjusting the corresponding keypoints. This code provides real-time visual feedback on screen, uses text-to-speech (espeak) audio output for immediate guidance, and it can counts valid squats automatically and displays the count on screen.

#  Prerequest

For how to connect and setup the AI camera, please refers to this document below
https://www.raspberrypi.com/documentation/accessories/camera.html#install-a-raspberry-pi-camera
https://www.raspberrypi.com/documentation/accessories/ai-camera.html#ai-camera
You can run the sample script to verify that the connection is correct. To try the sample post estimation script, run the following code:

    rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_posenet.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30



## AI Camera post estimation script

For pose estimation, we use higherhrnet_coco.rpk, one of the sample pose estimation models installed during AI Camera setup. You can find a list of models in the /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk directory.


For pose estimation code, we refer to the RaspberryPi Foundation's sample object recognition script. First, clone the script with the following command:

    git clone https://github.com/raspberrypi/picamera2.git

You can check the operation with the following command:
    
    python picamera2/examples/imx500_pose_estimation_higherhrnet_demo.py


After you have setup the prerequest, you can run the script by this comment
    
    python gym_pose_detect.py

