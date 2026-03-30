# Home Camera App

## Introduction
This sample application is also published in a [Qiita article](https://qiita.com/SSS-AtsushiNishimura/private/49058923518bf91729ce). Please refer to it as needed.

## Setup
Follow the installation instructions provided in the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/) to set it up.
Verify your uv installation by running:

```
uv --version
```

This application uses the [Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library). Follow link to find more on Application Module Library.

## Running the Sample Application
Open HomeCameraApp.py and modify the following 4 items:
```python
ALERT_SOUND_PATH = "/your/file/path/hoge.wav"
SCREENSHOT_FILENAME = "ScreenShot.png"
OAUTH_TOKEN = 'Slack OAuth Token'
CHANNEL_ID = 'Slack Channel ID'
```
After making the changes, save HomeCameraApp.py and then execute the command to start the application and install all the required packages:

```bash
uv run HomeCameraApp.py
```
## Notes
"Slack" is a service provided by a third party. We cannot be held responsible for any damages caused by the use of this service.  
Please confirm the service details on your own before use.