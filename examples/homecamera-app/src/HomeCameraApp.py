# SPDX-FileCopyrightText: 2025 Sony Semiconductor Solutions Corporation
#
# SPDX-License-Identifier: Apache-2.0

import time
import pygame
import cv2
import slack
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

# Definition of const
ALERT_SOUND_PATH = "/your/file/path/hoge.wav"
SCREENSHOT_FILENAME = "ScreenShot.png"
OAUTH_TOKEN = 'Slack OAuth Token'
CHANNEL_ID = 'Slack Channel ID'
PAUSE_AFTER_DETECTION = 3  # (seconds)

#Audio Output
def initialize_pygame():
    pygame.init()
    pygame.mixer.init()
    print("[INFO] Pygame initialized.")

def play_alert_sound():
    try:
        sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
        sound.play()
        print("[INFO] Alert sound played successfully.")
    except pygame.error as e:
        print(f"[ERROR] Failed to play alert sound: {e}")

# ScreenShot
def save_screenshot(filename, image):
    try:
        cv2.imwrite(filename, image)
        print(f"[INFO] Screenshot saved to {filename}.")
    except Exception as e:
        print(f"[ERROR] Failed to save screenshot: {e}")

#Slack Message
def send_slack_message(filename):
    client = slack.WebClient(token=OAUTH_TOKEN)
    try:
        response = client.files_upload_v2(
            channel=CHANNEL_ID,
            file=filename,
            title="Detect Person"
        )
        print("[INFO] Slack message sent successfully.")
    except SlackApiError as e:
        print(f"[ERROR] Failed to send Slack message: {e.response['error']}")


#Detection by modlib
def detect_person():
    print("[INFO] Starting person detection.")
    device = AiCamera()
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    last_detection_time = 0

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.55]
            labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
            names = [label.split(":")[0] for label in labels]

            annotator.annotate_boxes(frame, detections, labels=labels)
            frame.display()

            if "person" in names:
                current_time = time.time()
                if current_time - last_detection_time > PAUSE_AFTER_DETECTION:
                    print("[INFO] Detection action started.")
                    play_alert_sound()
                    save_screenshot(SCREENSHOT_FILENAME, frame.image)
                    send_slack_message(SCREENSHOT_FILENAME)
                    print("[INFO] Detection action completed.")
                    last_detection_time = time.time()


#main処理
if __name__ == "__main__":
    initialize_pygame()
    detect_person()