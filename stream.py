import cv2 
import platform
import subprocess as sp
import numpy as np
from pathlib import Path
from ultralytics import YOLO


#_________________Example URL______________________
#https://hd-auth.skylinewebcams.com/live.m3u8?a=megnas4do2fn4d43jqnpjp7fh6
#

VIDEO_URL = "https://hd-auth.skylinewebcams.com/live.m3u8?a=eu3scgop59mm73jhtvi49p1834" 

if platform.system() == "Windows": #Windows
    FFMPEG_BIN = Path("C:/Users/vlads/OneDrive/Desktop/ffmpeg/bin/ffmpeg.exe")
elif platform.system() == "Darwin": #MacOS
    FFMPEG_BIN = Path("/opt/homebrew/Cellar/ffmpeg/5.1.2_6/bin/ffmpeg")
else: #Linux
    FFMPEG_BIN = Path("insert path")

W = 1920
H = 1080

#Model loader
model1 = YOLO("yolov8l.pt") 
model2 = YOLO("yolov8l.pt") 

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
    raw_image = pipe.stdout.read(W*H*3) # read 432*240*3 bytes (= 1 frame)
    if len(raw_image) != W*H*3:
        print("Stream couldn't be loaded")
        break
    image = np.frombuffer(raw_image, np.uint8)
    image = image.reshape((H,W,3))
    #Apply models and plot them
    det1 = model1(image, device='mps') # mps for MacOS Sillicon, cuda for Nvidia GPU or cpu
    plot1 = det1[0].plot() 

    det2 = model2(image, device='mps') 
    plot2 = det2[0].plot() 

    cv2.imshow("First model", plot1)
    cv2.imshow("Second model", plot2)

    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()