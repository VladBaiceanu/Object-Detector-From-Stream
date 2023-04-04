import cv2 
import platform
import subprocess as sp
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO

'''
    Before running the code make sure:

    1) To install FFMPEG and add it's binary path below (lines    )
    
    2) Get the stream link; in the example given skylinewebcams.com creates a different session for every IP address
    => Access the network activity (easiest with inspect element) and get the link with .m3u8 extension

    3) Add the link to the "VIDEO_URL" variable (line )

    4) In order to run the yolov8 model properly, choose which device you are running the model on: Nvidia, CPU or Mac Sillicon

   __________________________Example Stream URL_____________________________
   https://hd-auth.skylinewebcams.com/live.m3u8?a=megnas4do2fn4d43jqnpjp7fh6
'''

# load the COCO class names
with open('models/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the SSD and yolov8 models
model = cv2.dnn.readNet(model='models/frozen_inference_graph.pb',
                        config='models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', #Using the mobilenetv2 single-shot detector
                        framework='TensorFlow')
model1 = YOLO("models/yolov8l.pt") 

#Insert here the stream URL and it's resolution: W - width, H - height
VIDEO_URL = "https://hd-auth.skylinewebcams.com/live.m3u8?a=eu3scgop59mm73jhtvi49p1834"
W = 1920 
H = 1080 

if platform.system() == "Windows": #Windows
    FFMPEG_BIN = Path("C:/Users/vlads/OneDrive/Desktop/ffmpeg/bin/ffmpeg.exe")
elif platform.system() == "Darwin": #MacOS
    FFMPEG_BIN = Path("/opt/homebrew/Cellar/ffmpeg/5.1.2_6/bin/ffmpeg")
else: #Linux
    FFMPEG_BIN = Path("insert path")

#Load stream
pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
            stdin = sp.PIPE, stdout = sp.PIPE)

while True:
    #Preprocces frame from stream
    raw_image = pipe.stdout.read(W*H*3) # read 1920*1080*3 bytes -> 1 frame
    if len(raw_image) != W*H*3:
        print("Stream couldn't be loaded")
        break
    image = np.frombuffer(raw_image, np.uint8)
    image = image.reshape((H,W,3))

    if image.any():
        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                     swapRB=True)
        #Apply models and plot them
        model.setInput(blob)
        output = model.forward()        
        det1 = model1(image, device='mps') # mps for MacOS Sillicon, cuda for Nvidia GPU or cpu
        plot1 = det1[0].plot()

        # loop over each of the detections for the DNN model
        for detection in output[0, 0, :, :]:
            # get the confidence
            confidence = detection[2]
            if confidence > 0.08: #draw the bounding boxes only if the confidence is above 8%
                class_id = detection[1]
                # map the class id to the class 
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                # get the bounding box coordinates
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                # get the bounding box width and height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height

                # draw the boxes and give a it's class name to it
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
        cv2.imshow('SSD Mobilenetv2', image)
        cv2.imshow("Yolo v8 ", plot1)

    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()