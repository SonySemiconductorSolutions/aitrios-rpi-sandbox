#
# Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time
from collections import deque

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

# Initialize Flask application
app = Flask(__name__)

# Global variables
frame_buffer = None
frame_lock = threading.Lock()
processing_active = False
alert_active = False

# Keypoint names and corresponding indices
KEYPOINT_NAME = [
    "nose",             #0
    "leftEye",          #1
    "rightEye",         #2
    "leftEar",          #3
    "rightEar",         #4
    "leftShoulder",     #5
    "rightShoulder",    #6
    "leftElbow",        #7
    "rightElbow",       #8
    "leftWrist",        #9
    "rightWrist",       #10
    "leftHip",          #11
    "rightHip",         #12
    "leftKnee",         #13
    "rightKnee",        #14
    "leftAnkle",        #15
    "rightAnkle"        #16
]

# Keypoint indices for motion detection (both wrists and both knees)
MOTION_CHECK_KEYPOINTS = [9, 10, 13, 14]  # leftWrist, rightWrist, leftKnee, rightKnee

# Monitoring area and motion detection settings (hardcoded in the code)
monitoring_area = {'x': 100, 'y': 100, 'width': 300, 'height': 300}
motion_settings = {'time_window': 6, 'movement_threshold': 10000, 'check_interval': 1}

# Person tracking data
class PersonTracker:
    def __init__(self):
        self.positions = deque(maxlen=100)  # Past position data
        self.in_area = False  # Whether a person is in the area
        self.last_check_time = time.time()
        self.last_movement = 0  # Last calculated movement amount

person_tracker = PersonTracker()

# Camera and model initialization function
def initialize_camera():
    device = AiCamera()
    model = Posenet()
    device.deploy(model)
    return device, Annotator()

# Background task for processing the camera stream
def process_camera_stream():
    global frame_buffer, processing_active, alert_active, person_tracker
    
    device, annotator = initialize_camera()
    
    with device as stream:
        while processing_active:
            for frame in stream:
                # Skeleton detection
                detections = frame.detections[frame.detections.confidence > 0.5]
                
                # Get image size
                h, w = frame.image.shape[:2]
                
                # Process detected skeleton data
                if len(detections) > 0:
                    # Use the first detection result
                    detection = detections[0]
                    
                    # Get keypoints
                    keypoints = []
                    
                    # Keypoints are a 1D array with alternating x and y coordinates
                    # [y0, x0, y1, x1, y2, x2, ...]
                    kp_array = detection.keypoints[0]
                    
                    # Calculate the number of keypoints (array length / 2)
                    num_keypoints = len(kp_array)
                    
                    for i in range(num_keypoints):
                        # Get y and x coordinates
                        y = kp_array[i][1]
                        x = kp_array[i][0]
                        
                        keypoints.append({
                            'id': i,
                            'name': KEYPOINT_NAME[i] if i < len(KEYPOINT_NAME) else f"kp_{i}",
                            'x': float(x * w),
                            'y': float(y * h)
                        })
                    
                    # Calculate the center point of the person
                    center_point = get_center_point(keypoints)
                    
                    # Check if the person is in the monitoring area
                    is_in_area = is_point_in_area(center_point, monitoring_area)
                    
                    # If the state of being in the area changes
                    if is_in_area != person_tracker.in_area:
                        person_tracker.in_area = is_in_area
                        if not is_in_area:
                            alert_active = False  # Clear alert when leaving the area
                    
                    # Record position data only if the person is in the area
                    if is_in_area:
                        # Add new data
                        person_tracker.positions.append({
                            'timestamp': time.time(),
                            'keypoints': keypoints,
                            'frame_size': (w, h)
                        })
                        
                        # Periodically check motion
                        current_time = time.time()
                        if current_time - person_tracker.last_check_time >= motion_settings['check_interval']:
                            check_movement()
                            person_tracker.last_check_time = current_time
                else:
                    # If no person is detected
                    if person_tracker.in_area:
                        person_tracker.in_area = False
                        alert_active = False  # Clear alert when no person is detected
                
                # Draw the monitoring area
                cv2.rectangle(
                    frame.image,
                    (monitoring_area['x'], monitoring_area['y']),
                    (monitoring_area['x'] + monitoring_area['width'], monitoring_area['y'] + monitoring_area['height']),
                    (0, 255, 0) if not alert_active else (0, 0, 255),
                    2
                )
                
                # Draw alert state
                if alert_active:
                    # Display a red band at the top
                    cv2.rectangle(
                        frame.image,
                        (0, 0),
                        (frame.image.shape[1], 50),
                        (0, 0, 255),
                        -1
                    )
                    # Warning text
                    cv2.putText(
                        frame.image,
                        "ALERT: No Motion Detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                # Display the amount of motion
                if person_tracker.in_area:
                    cv2.putText(
                        frame.image,
                        f"Movement Amount: {person_tracker.last_movement:.2f}",
                        (10, frame.image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )
                
                # Annotate skeleton
                annotator.annotate_keypoints(frame, detections)
                
                # Convert the frame to JPEG format
                _, buffer = cv2.imencode('.jpg', frame.image)
                
                # Update the buffer
                with frame_lock:
                    frame_buffer = buffer.tobytes()
                
                # Add a short wait time to reduce load
                time.sleep(0.01)
                break  # Exit the loop after processing one frame

# Function to calculate the center point of a person
def get_center_point(keypoints):
    valid_points = [kp for kp in keypoints if kp['x'] > 0.0 and kp['y'] > 0.0]
    if not valid_points:
        return None
    sum_x = sum(kp['x'] for kp in valid_points)
    sum_y = sum(kp['y'] for kp in valid_points)
    return {'x': sum_x / len(valid_points), 'y': sum_y / len(valid_points)}

# Function to check if a point is in the monitoring area
def is_point_in_area(point, area):
    if not point:
        return False
    return (point['x'] >= area['x'] and point['x'] <= area['x'] + area['width'] and
            point['y'] >= area['y'] and point['y'] <= area['y'] + area['height'])

# Function to check motion
def check_movement():
    global person_tracker, alert_active
    
    # Do not check if no person is in the area
    if not person_tracker.in_area or len(person_tracker.positions) < 2:
        return
    
    current_time = time.time()
    
    # Get data within a certain time window
    cutoff_time = current_time - motion_settings['time_window']
    valid_positions = [p for p in person_tracker.positions if p['timestamp'] >= cutoff_time]
    
    # Check if there is enough data
    if len(valid_positions) >= 2:
        # Calculate the amount of motion
        movement_amount = calculate_movement_specific_keypoints(valid_positions)
        person_tracker.last_movement = movement_amount
        
        # Trigger alert if below the threshold
        if movement_amount < motion_settings['movement_threshold']:
            alert_active = True
            print("No Motion Detected:", movement_amount, "<", motion_settings['movement_threshold'])
        else:
            # Clear alert if there is enough motion
            alert_active = False
            print("Motion Detected   :", movement_amount, ">", motion_settings['movement_threshold'])

# Function to calculate motion for specific keypoints (both wrists and both knees)
def calculate_movement_specific_keypoints(positions):
    # Get the oldest and newest positions
    oldest = positions[0]
    newest = positions[-1]
    
    # Get frame size
    w, h = oldest['frame_size']
    
    total_distance = 0
    valid_keypoint_count = 0
    
    # Check only specific keypoints (both wrists and both knees)
    for keypoint_idx in MOTION_CHECK_KEYPOINTS:
        # Find corresponding keypoints in the old and new frames
        old_kp = next((kp for kp in oldest['keypoints'] if kp['id'] == keypoint_idx), None)
        new_kp = next((kp for kp in newest['keypoints'] if kp['id'] == keypoint_idx), None)
        
        # If keypoints are detected in both frames
        if old_kp and new_kp:
            # Convert to pixel coordinates
            x0 = int(old_kp['x'] * w)
            y0 = int(old_kp['y'] * h)
            x1 = int(new_kp['x'] * w)
            y1 = int(new_kp['y'] * h)
            
            # Calculate squared distance (Euclidean distance squared)
            distance = (x1 - x0)**2 + (y1 - y0)**2
            
            total_distance += np.sqrt(distance)  # Take square root for actual distance
            valid_keypoint_count += 1
    
    # Return average distance if there are valid keypoints, otherwise 0
    if valid_keypoint_count > 0:
        return total_distance / valid_keypoint_count
    else:
        return 0

# Video stream generator function
def generate_frames():
    global frame_buffer
    
    while True:
        # Wait until the frame buffer is available
        if frame_buffer is not None:
            with frame_lock:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        
        # Short wait time
        time.sleep(0.03)

# Root page
@app.route('/')
def index():
    return render_template('index.html')

# Video stream endpoint
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start camera processing endpoint
@app.route('/start')
def start_processing():
    global processing_active
    
    if not processing_active:
        processing_active = True
        threading.Thread(target=process_camera_stream, daemon=True).start()
        return jsonify({"status": "started"})
    
    return jsonify({"status": "already running"})

# Stop camera processing endpoint
@app.route('/stop')
def stop_processing():
    global processing_active, alert_active
    
    processing_active = False
    alert_active = False
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
