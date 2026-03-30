# Virtual Fitting Application Using Raspberry Pi AI Camera

This is an application that detects human poses with the Raspberry Pi AI Camera and overlays T-shirt images on them.
<img src="virtual-fitting_sample.gif" alt="GIF image of Virtual Fitting App" width="700">  

Additional explanation of this application can be found in a [Qiita article](https://qiita.com/SSS-TakahashiKatsuya/items/a4177ee6808e45a47175) (Japanese).


## Setup
To set up an environment for the application, UV is used for the project
### Install UV
After cloning this repository using git clone, Follow the installation instructions provided in the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/) to set it up.
Verify your uv installation by running:

```
uv --version
```

This application uses the [Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library). Follow link to find more on Application Module Library.

### Prepare a Image to Overlay
Prepare a transparent PNG image (RGBA) or normal image (RGB) and name it as "tshirt.png", and place it in the same directory as app.py".  
Transparent PNG T-shirt images can be downloaded from [this 3rd party link](https://www.irasutoya.com/2018/07/t.html) (Please use this at your own responsibility) or use your own image.


## Running the Application
Navigate to the application folder, use uv to run the application to install the pyproject.toml and start the application:
```shell
uv run app.py
```
## Model
The Model used in this application is Posenet Model that detects keypoints.  
You can find the converted model for Raspberry Pi AI Camera from [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_posenet.rpk).
