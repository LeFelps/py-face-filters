#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from math import dist
from cv2 import log
import numpy as np
import dlib

# inicializa o detector e preditor do dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# inicializar vídeo
# caso não tenha webcam escolha um video de teste .mp4.
cap = cv2.VideoCapture(0)

title = "Filtros"
screen = 0

def mouse_click(event, x, y, flags, param):
    global screen
    if event == cv2.EVENT_LBUTTONDOWN:
        if screen == 3:
            screen = 0
        else:
            screen += 1
    if event == cv2.EVENT_RBUTTONDOWN:
        if screen == 0:
            screen = 3
        else:
            screen -= 1

cv2.namedWindow(title)
cv2.setMouseCallback(title, mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if screen == 0: 
        output = frame

    if screen == 1: 
        output = blur_mask

    if screen == 2: 
        output = rainbow_filter
        
    if screen == 3: 
        output = frame
            
            
    # detectar faces (grayscale)
    rects = detector(gray, 0)

    def getCenter(x1,x2,y1,y2):
        return (round(x1 + ((x2 - x1)/2)), round(y1 + ((y2 - y1)/2)))

    # loop nas detecções de faces
    for rect in rects:

        shape = predictor(gray, rect)
        coords = np.zeros((shape.num_parts, 2), dtype=int)
        for i in range(0,68): 
            coords[i] = (shape.part(i).x, shape.part(i).y)



        #FILTER 1
        x, y = rect.left(), rect.top()  # topo esquerda
        x1, y1 = rect.right(), rect.bottom()  # baixo direita

        h, w = frame.shape[:2]

        img_temp = cv2.resize(frame, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
        blurred_img = cv2.resize(img_temp, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask = cv2.circle(mask, getCenter(x,x1,y,y1), round(max([(x1-x),(y-y1)])/2 + 30), [255, 255, 255], -1)

        blur_mask = np.where(mask == np.array([255, 255, 255]), blurred_img, frame)



        #FILTER 2
        rainbow_filter = cap.read()[1]

        rainbow = cv2.imread("./img/rainbow.png")

        h, w = rainbow.shape[:2]

        rainbow = cv2.resize(rainbow, (round(w/2), round(h/2)), interpolation=cv2.INTER_LINEAR)

        M_x1, M_y1 = coords[49]
        M_x2, M_y2 = coords[55]

        x = round(M_x1 + ((M_x2 - M_x1)/2))
        y = round(M_y1 + ((M_y2 - M_y1)/2))

        rainbowH, rainbowW = rainbow.shape[:2]

        x_offset = x - round(rainbowW/2)
        y_offset = y - round(rainbowW/2)

        x_end = x_offset + rainbowW
        y_end = y_offset + rainbowH  

        rainbow_filter[y_offset:y_end,x_offset:x_end] = rainbow

        

        # FILTER 3


    # Exibe resultado
    cv2.imshow(title, output)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()
