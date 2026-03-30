# Privacy Mask Application Using Raspberry Pi AI Camera
## Overview
This example is an application that detects people using object detection with the Raspberry Pi AI Camera and applies privacy masks to them.  
<img src="img/PrivacyMask_Result.gif" alt="result.gif" width="700">  

This sample application is also published in a [Qiita article](https://github.com/SonySemiconductorSolutions/aitrios-qiita/blob/feature/QP-69-PrivacyMask/articles/20250128_PrivacyMaskApp_withRaspi/PrivacyMask.md) for your reference as needed.


## Setup
Follow the installation instructions provided in the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/) to set it up.
Verify your uv installation by running:

```
uv --version
```

This application uses the [Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library). Follow link to find more on Application Module Library.

### Running the Application
 
Then navigate to the application folder and execute the following command to start the application and install all required packages:

```bash
uv run main.py
```