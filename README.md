# Object detector using two different models
## Main ideas:
- The program captures in real time a video stream from a public camera
- Video frames are captured by the FFmpeg video handler and processed by two different object detection models: *Yolov8* and *SSD MobileNetv2*
- The purpose is to compare the performance of the models in the same conditions

 <img width="1028" alt="Screenshot 2023-04-04 at 05 42 04" src="https://user-images.githubusercontent.com/92524259/229673829-b479dd9c-eed2-4737-970e-8ec6be9dc6b5.png">
 <img width="1028" alt="Screenshot 2023-04-04 at 05 41 57" src="https://user-images.githubusercontent.com/92524259/229673849-3cb5b074-f504-43b9-bf09-cdb74ba47718.png">
 
## Stream capturing
  The program is made to capture streams using the _.m3u8_ extension. The address of the stream can be found using inspect element, network activity. However, the best solution would be capturing a _RTSP_ stream but there are few publicly available, so having your own public IP camera is ideal.

## Prerequisites
- At least Python 3.8
- FFmpeg installed
- Best results are on CUDA enabled GPU's but it should work on Apple Sillicon or CPU's
>**_NOTE:_** Necessary libraries can be installed using the command:
>``pip install -r requirements.txt``
