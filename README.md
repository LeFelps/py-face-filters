# py-face-filters

Uses image recognition to detect landmarks on a face and apply a few filters

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
