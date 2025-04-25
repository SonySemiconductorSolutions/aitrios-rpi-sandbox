＃ readme


In this article, I will introduce a method for using 


This article explores the use of an object detection model to detect whether a driver is using a mobile phone. Specifically, we utilize the object detection model "nanodet_plus_416x416_pp" from Model Zoo to identify  the mobile phone. If a mobile phone is detected, an alarm sound and an email notification will be triggered to warn that the driver may be using their phone.


1, First update

    $ sudo apt update && sudo apt full-upgrade 

2, Install the imx500

    $ sudo apt install imx500-all

3, confirm the Linux Kernel version

    uname -a
    $ Linux raspberrypi 6.6.62+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.62-1+rpt1 (2024-11-25) aarch64 GNU/Linux
    
4, Clone github and run the script

    $ git clone <https://github.com/raspberrypi/picamera2>
    $ cd ~/picamera2/examples/driver_detect
    $ python driver_detect.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk


- When detecting the cell phone
Cell phone usage time will be informed and the alarm with using espeak: ("Warning! Warning! Cell phone detected! Cell phone detected! Please do not use cell phone during driving")

Be noticed that please revise the email address and the path for saving screeningshot before you run the script.
