
<span>
  <img src='GIF_long.gif' width="888" height="500"> 
  <img src='GIF_short.gif' width="888" height="500">
</span>

# Traffic Analysis System

The aim of this project is to develop a cheap and easy-to-deploy system of analysing traffic in some road by using low-resource and small devices (RespberryPi, JetsonNano) accelerated with USB Coral TPU to work on real time.

By giving this system a video (or using the camera (*in the future*)) you'll get a json file with all the vehicles crossing a detection line defined by the user. This file will not only contain the number of vehicles detected, it will contain information about every detection regarding the type of vehicle, the time when it was detected and an estimation of its speed (*still not accurate, work in progress*).

The system detects 4 kinds of vehicles: **Car, Bus, Truck** and **Motorbike**.

There's an example in this repository: *vehicles.json* .

No images or other sensible information is saved about the vehicle. The goal of this system is to extract traffic information for analysis at low cost and nothing else.


## Hardware
<b>Primary Components</b>
1) Raspberry Pi (<a target="_blank" href="https://www.raspberrypi.org/products/raspberry-pi-4-model-b"/>Link</a>)
2) Raspberry Pi camera (<a target="_blank" href="https://www.raspberrypi.org/products/camera-module-v2/">Link</a>)
3) JetsonNano (<a target="_blank" href="https://developer.nvidia.com/embedded/jetson-nano-developer-kit">Alternative to Rpi</a>)
4) Google Coral Accelerator (<a target="_blank" href="https://coral.ai/products/accelerator">Strongly recommended</a>)

## Software

### Install environment

First, I strongly recommend following the instructions in the below GitHub repository to install all needed libraries. 
It is a really well done and detailed step by step guide and more importantly the one I followed so it will give you less problems if you stick to it:
```
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
```
### Download this repository and run!

Clone this repository on your own machine.

```
git clone https://github.com/smorantg2/tfm_traffic
```
Once you have all the files in your desired folder, run the main script "EmbosNet.py" with the following command to get the traffic data from a video.

```
python3 EmbosNet.py --model_name detect.tflite --model_path embosnet_v3_quant/model/ --video <PATH_TO_VIDEO> --output <PATH_TO_SAVE_OUTPUT> --threshold 0.51 --use_tpu True --display True
```
(IMPORTANT!)
The script will ask you first to draw the detection line with 2 double-clicks and the perspective of the road with 4 double-clicks in the following order:
1. Top Left 
2. Top Right 
3. Bottom Left 
4. Bottom Right

There are some arguments you can change depending on whether you want to use TPU or not, see the process or save the output video (like the GIFs above).

```
--model_name,   required=True,                            help = Name of the .tflite file, if different than detect.tflite
--model_path,   required=True,                            help = Folder the .tflite file is located in
--video,        required = True, type=str,                help = "path to input video file
--output,       type=str,                                 help = path to optional output video file, with "/"
--threshold,    required = True, type=float, default=0.5, help= minimum probability to filter weak detections
--use_tpu,      required =True,                           help= Whether to use TPU or not
--display       type = bool,                              help= Whether to display all the action or not
```

Once you've the script finished you'll get a json file in the same folder you have the main script (EmbosNet.py) with all the detections' information.

## Model Information

The model used in this repository is a MobileNet_V2 quantized so it can provide really fast predictions. 

The model was trained with a custom dataset developed from zero by me for this particular project. 
The amount of data is, as you can imagine, ridiculously small. However, the model behaves better than expected and well enough for the purpose of showing the main idea of this project.

Other architectures and datasets could be used and the accuracy of the system may improve significantly.



--------------------------------------------------------------------------------------------------------------------------------

## References

The following article was really helpful and was referenced several times during the project:

```
https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
```

The following repo was referenced during data collection and model development:
```
https://github.com/nathanrooy/rpi-urban-mobility-tracker
```
