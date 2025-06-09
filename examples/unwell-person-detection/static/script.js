//
// Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

document.addEventListener('DOMContentLoaded', function() {
    const videoStream = document.getElementById('video-stream');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const detectionInfo = document.getElementById('detection-info');
    
    // Disable stop button initially
    stopBtn.disabled = true;

    // Error handling
    videoStream.onerror = function() {
        detectionInfo.innerHTML = '<p class="error">Failed to load video stream</p>';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    };

    // Start button click event
    startBtn.addEventListener('click', function() {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        fetch('/start')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log('Processing started:', data);
                videoStream.src = '/video_feed?' + new Date().getTime(); // Avoid cache
                videoStream.style.display = 'block';
                detectionInfo.innerHTML = '<p>Pose detection started</p>';
            })
            .catch(error => {
                console.error('Error:', error);
                detectionInfo.innerHTML = '<p class="error">An error occurred</p>';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            });
    });

    // Stop button click event
    stopBtn.addEventListener('click', function() {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        
        fetch('/stop')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log('Processing stopped:', data);
                videoStream.style.display = 'none';
                detectionInfo.innerHTML = '<p>Pose detection stopped</p>';
            })
            .catch(error => {
                console.error('Error:', error);
                startBtn.disabled = true;
                stopBtn.disabled = false;
            });
    });
});
