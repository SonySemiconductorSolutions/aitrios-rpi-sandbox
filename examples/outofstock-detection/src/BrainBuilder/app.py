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
import time
from functools import lru_cache

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

import numpy as np
from collections import deque
import threading
import schedule
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

# Email settings
delivery_time = "18:00"
sender = "Sender's_Email_Address (gmail)"
password = "Your_own_Password"  # For Gmail, use app password
receiver = "Recipient's_Email_Address"
# AI setting
last_detections = []
detection_label = "Egg"
num_frames = 30 # Number of frames to retain
check_detections = deque(maxlen=num_frames) # Store detection data for the past num_frames frames
latest_frame = 0 # Store the average pixel value of the latest frame (for calculation)
current_status = None # Status used for notifications
img_threshold = 45 # Threshold for average pixel value

def check_outofstack():
    """
    Check if objects are out of stock based on detection history.
    Updates status when stock level changes are detected.
    """
    global current_status
    
    if latest_frame > img_threshold:
        Append_last_detections()
    else:
        if len(check_detections) != 0:
            current_status = max(sublist.count(detection_label) for sublist in check_detections)
                
# Check for person in detection area
def Append_last_detections():    
    """
    Append detected objects to the detection history queue.
    Extracts labels from current detections and adds them to the history.
    """
    global check_detections
    tmp_list = []
    detections = last_detections
    labels = get_labels()
    if detections == []:
        check_detections.append([])
        return 
    else:
        for detection in detections:
            tmp_list.append(labels[int(detection.category)])
        check_detections.append(tmp_list)
        return

def send_email():
    """
    Send email notification about current inventory status.
    Configures email content based on the current_status variable.
    """    
    # Create email content
    if current_status == None:
        subject = "AICamera_demo_Inventory Notification"
        body = "Inventory status not detected"
    elif current_status == 0:
        subject = "AICamera_demo_Inventory Notification"
        body = "Out of stock!! Let's buy some on the way home!!!!"
    else:
        subject = "AICamera_demo_Inventory Notification"
        body = "Remaining inventory: " + str(current_status) + " items"
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg['Date'] = formatdate()
    
    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"Email sent successfully: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error occurred: {e}")

def schedule_task():
    while True:
        schedule.run_pending()
        time.sleep(1)


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def check_outofstack_and_parse_and_draw_detections(request):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    detections = parse_detections(request.get_metadata())
    draw_detections(request, detections)
    check_outofstack()


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_detections
    threshold = args.threshold

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score >= threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    with open(args.labels, "r") as f:
        labels = f.read().split("\n")

    return labels


def draw_detections(request, detections, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    global latest_frame
    labels = get_labels()
    with MappedArray(request, stream) as m:
        latest_frame = np.mean(m.array[:,:,:3])
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            cv2.putText(
                m.array,
                label,
                (x + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 255, 0))

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="Path of the model")
    parser.add_argument("--model", type=str, default="network.rpk", help="Path of the model")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument(
        "--labels",
        type=str,
        # default="assets/labels.txt",
        default="labels.txt",
        help="Path to the labels file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)

    # Create and start email delivery thread
    schedule.every().day.at(delivery_time).do(send_email)
    thread = threading.Thread(target=schedule_task)
    thread.daemon = True
    thread.start()
    
    picam2 = Picamera2()

    imx500.show_network_fw_progress_bar()
    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=28)
    picam2.start(config, show_preview=True)
    picam2.pre_callback = check_outofstack_and_parse_and_draw_detections

    while True:
        time.sleep(0.5)
