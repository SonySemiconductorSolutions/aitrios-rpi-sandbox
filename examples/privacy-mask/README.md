# Privacy Mask Application Using Raspberry Pi AI Camera
## Overview
This example is an application that detects people using object detection with the Raspberry Pi AI Camera and applies privacy masks to them.  
<img src="img/PrivacyMask_Result.gif" alt="result.gif" width="700">  

This sample application is also published in a [Qiita article](https://github.com/SonySemiconductorSolutions/aitrios-qiita/blob/feature/QP-69-PrivacyMask/articles/20250128_PrivacyMaskApp_withRaspi/PrivacyMask.md) for your reference as needed.


## Setup
This application utilizes the Application Module Library as a submodule.
### Updating Submodules
After cloning the repository using git clone, incorporate the submodule using the following command:
```shell
git submodule init
git submodule update
```
For subsequent Application Module Library setup instructions, please refer to [this link](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library).

## Running the Application
Navigate to the src folder and execute the following command:

    python main.py