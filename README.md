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


### Important sidenote
Both the **landmarks.dat** file and the **video.mp4** file (if used) must be on the **same root folder** along with the **filters.py** file
