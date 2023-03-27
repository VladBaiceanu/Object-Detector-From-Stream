import cv2 
import subprocess as sp
import numpy as np
from pathlib import Path


#_________________URL-URI VALIDE______________________
#https://hd-auth.skylinewebcams.com/live.m3u8?a=megnas4do2fn4d43jqnpjp7fh6
#

VIDEO_URL = "https://hd-auth.skylinewebcams.com/live.m3u8?a=g7vn08dmv4bovag2kvtblke1q4"
#FFMPEG_BIN = Path("C:/Users/vlads/OneDrive/Desktop/ffmpeg/bin/ffmpeg.exe")  WINDOWS
FFMPEG_BIN = Path("/opt/homebrew/Cellar/ffmpeg/5.1.2_6/bin/ffmpeg")
W = 1920
H = 1080
print("Imi merge path-ul:",FFMPEG_BIN.exists())

cv2.namedWindow("Detection")
car_cascade = cv2.CascadeClassifier('cars.xml')

pipe = sp.Popen([ FFMPEG_BIN, "-i", VIDEO_URL,
           "-loglevel", "quiet", # no text output
           "-an",   # disable audio
           "-f", "image2pipe",
           "-pix_fmt", "bgr24",
           "-vcodec", "rawvideo", "-"],
            stdin = sp.PIPE, stdout = sp.PIPE)
while True:
    raw_image = pipe.stdout.read(W*H*3) # read 432*240*3 bytes (= 1 frame)
    if len(raw_image) != W*H*3:
        print("raw_image != W*H*3")
        break
    image =  np.fromstring(raw_image, dtype='uint8')
    image = np.frombuffer(raw_image, np.uint8)
    image = image.reshape((H,W,3))
    #cv2.imshow("Detection",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Detection",image)
    
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()