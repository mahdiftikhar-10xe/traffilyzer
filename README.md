# Description
"traffilyzer" is a smart traffic analysis app based on YOLOv8.
It is capable of detecting, classifying, tracking and counting
vehicles passing through user-defined regions.\
The YOLOv8 model used is based on
[VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
dataset developed by the AISKYEYE team at Lab of
Machine Learning and Data Mining, Tianjin University, China.\
Pre-trained models can be found at:
* [yolov8n-visdrone](https://huggingface.co/mshamrai/yolov8n-visdrone)
* [yolov8s-visdrone](https://huggingface.co/mshamrai/yolov8s-visdrone)
* [yolov8m-visdrone](https://huggingface.co/mshamrai/yolov8m-visdrone)
* [yolov8l-visdrone](https://huggingface.co/mshamrai/yolov8l-visdrone)
* [yolov8x-visdrone](https://huggingface.co/mshamrai/yolov8x-visdrone)

## Installation
Start by cloning the repository:
```
git clone https://github.com/sk8thing/traffilyzer.git
```
Install the requirements, I also suggest creating a virtual environment:
```
cd traffilyzer
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```
Start the application:
```
main.py [-h] --model MODEL [--conf CONF] [--iou IOU] --video VIDEO
```
## How to use
The application will try to use hardware acceleration if possible.
Check top-left corner for information about the device used.\
Once the application is running you can press P to pause and Q to exit.\
While paused you can left or right click 4 points to create a counting region.
Left click will create input regions and right click will create output regions.
After defining the regions you can unpause. The number displayed in the center
of each region (the one matching the color of the region outline) is the total
number of detections counted, output regions will also display the number of
detections from each input zone.\
After exiting the application will generate an Excel file with statistics.
## Screenshots
![1](https://github.com/sk8thing/traffilyzer/assets/101511232/514f0317-dea1-4544-9860-2f55fec2bc63)
![2](https://github.com/sk8thing/traffilyzer/assets/101511232/3d8880b3-9aa8-49f9-91b9-c5d5f1eea67d)