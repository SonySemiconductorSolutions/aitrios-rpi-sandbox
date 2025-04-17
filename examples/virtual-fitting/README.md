# Virtual Fitting Application Using Raspberry Pi AI Camera
## Overview
This is an application that detects human poses with the Raspberry Pi AI Camera and overlays T-shirt images on them.
<img src="virtual-fitting_sample.gif" alt="GIF image of Virtual Fitting App" width="700">  

Additional explanation of this application can be found in a [Qiita article](https://github.com/SonySemiconductorSolutions/aitrios-qiita/) (Japanese).


## Setup
This application uses the [Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library) as a submodule.
### Install Submodule
After cloning this repository using git clone, incorporate the submodule using the following commands:
```shell
git submodule init
git submodule update
```
For subsequent submodule installation instructions, please refer to [this link](https://qiita.com/organizations/AITRIOSbySony_JP/).

### Prepare a Image to Overlay
Prepare a transparent PNG image (RGBA) and name it as "tshirt.png", and place it in the same directory as "tshirt.png.sample".  
Transparent PNG T-shirt images can be downloaded from [this 3rd party link](https://www.irasutoya.com/2018/07/t.html).  (Please use this at your own responsibility.)


## Running the Application
Navigate to the src folder and execute the following command:
```shell
    python3 app.py
```
## Model
The Model used in this application is Higherhrnet Model that detects both boundary boxes and keypoints.  
You can find the converted model for Raspberry Pi AI Camera from [this link](https://github.com/raspberrypi/imx500-models/blob/main/imx500_network_higherhrnet_coco.rpk).