# Unwell Person Detection Application
This application detects unwell individuals in a sport gym environment using the Raspberry Pi AI Camera.

![Unwell Person Detection](unwell-person-detection.png)

## Prerequisites

- A Raspberry Pi equipped with the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/)
- Python 3 installed on the Raspberry Pi

## Setup Instructions

1. Navigate to the example directory:
    ```bash
    cd /home/pi/aitrios-rpi-sandbox-dev/examples/unwell-person-detection
    ```

2. Install the [Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library):

    First, initialize and update the submodule:
    ```bash
    git submodule init
    git submodule update
    ```

    Then, follow the instructions provided [here](https://www.raspberrypi.com/products/ai-camera/) to set up the environment and build the required Python wheel.

3. Install additional dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the application:
    ```bash
    python3 app.py
    ```

2. Access the web interface in your browser at:  
   `http://<your_raspberrypi_ip>:5000/`
