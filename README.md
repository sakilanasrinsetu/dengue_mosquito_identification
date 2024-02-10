
### **Step-1: #For Install Python 3.9**

* conda create -n tf2 python=3.9
* conda activate tf2
* conda install protobuf or pip install protobuf

***Note*** Must be in Virtual enviroment

##### **git clone https://github.com/tensorflow/models.git**

### **Step-2: #Add Protobuf File (use_protobuf.py) in models/research Folder -And Add:**

>
>         import os
>         import sys
>         args = sys.argv
>         directory = args[1]
>         protoc_path = args[2]
>         for file in os.listdir(directory):
>             if file.endswith(".proto"):
>                 os.system(protoc_path+" "+directory+"/"+file+" --python_out=.")

### **Step-3: Protoc Command Setup**

>     python use_protobuf.py object_detection/protos protoc

### Step-4: Copy the Setup File From object_detection/packages/tf2 to research Folder

### Step-5: Install Setpup File

>     python -m pip install .
>     conda install numpy

Step-6: Checking Everything is ok or not:
   > python object_detection/builders/model_builder_tf2_test.py 
# Noted that everything from the Research Folder

Step-7: Install wget
    pip install wget
    python model_downloader.py

    python detect_from_image.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model -l data/mscoco_label_map.pbtxt -i test_images/

    python detect_from_image.py -m efficientdet_d0_coco17_tpu-32/saved_model -l labelmap.pbtxt -i test_images/dengue/ ## for my model
    python detect_from_image.py -m training/ -l labelmap.pbtxt -i test_images/dengue/

Doen the 1ST Step

Step2.1: Install Requitrment:
    pip install opencv-contrib-python

Step 2.2: Object Detaiction By Webcam:
    python detect_from_webcam.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/ -l data/mscoco_label_map.pbtxt

    python detect_from_webcam.py -m efficientdet_d0_coco17_tpu-32/saved_model/ -l labelmap.pbtxt

Step 2.3: Object Detaiction By Video:
    python detect_from_video.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/ -l data/mscoco_label_map.pbtxt -v video/test.mp4

    python detect_from_video.py -m efficientdet_d0_coco17_tpu-32/saved_model/ -l labelmap.pbtxt -v video/dengue.mp4

python model_main_tf2.py --pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr - Working but error
python model_main_tf2.py --pipeline_config_path=training/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr- also workikg with error
