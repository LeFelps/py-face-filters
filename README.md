# Python face filters

Uses image recognition to detect landmarks on a face and apply a few filters

## How to use
After launching the program you can change the filter by pressing **mouse left button (next)** or **mouse right button (previous)**

## Dependencies
- OpenCv
- dlib
```
pip install opencv-python
pip install dlib
```

Uses the shape_predictor_68_face_landmarks.dat
from https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat


### Important sidenotes
1. Both the **landmarks.dat** file and the **video.mp4** file (if used) must be on the **same root folder** along with the **filters.py** file

2. To change from camera to video the following lines must be changed on lines 13 - 15:

from...
```
# starting video / opening webcam
cap = cv2.VideoCapture("video.mp4")
# cap = cv2.VideoCapture(0)
```
to...
```
# starting video / opening webcam
cap = cv2.VideoCapture("video.mp4")
# cap = cv2.VideoCapture(0)
```
where "video.mp4" is the name of the file between quotation marks
