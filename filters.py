#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
from math import dist
from cv2 import log
from cv2 import ROTATE_180
from cv2 import bitwise_and
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

def rotate_image(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
    
# function to handle mouse clicks (left = next, right = previous)
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
        output = overlap_filter
        
    if screen == 3: 
        output = custom_filter
            
            
    # detectar faces (grayscale)
    rects = detector(gray, 0)

    def getCenter(x1,x2,y1,y2):
        return (round(x1 + ((x2 - x1)/2)), round(y1 + ((y2 - y1)/2)))

    # loop nas detecções de faces
    if len(rects) > 0:
        for rect in rects:

            shape = predictor(gray, rect)
            # # displaying dots and indexes of the face landmarks on the frame
            # for i in range(1,68): 
            #     cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,255,0), thickness=-1)
            #     cv2.putText(frame, str(i), (shape.part(i).x,shape.part(i).y), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.3, color=(0, 0, 255))
            
            coords = np.zeros((shape.num_parts, 2), dtype=int)
            for i in range(0,68): 
                coords[i] = (shape.part(i).x, shape.part(i).y)



            #FILTER 1 - pixelated face
            x, y = rect.left(), rect.top()  # top left
            x1, y1 = rect.right(), rect.bottom()  # bottom right

            # getting image size
            h, w = frame.shape[:2]

            # creating a blurred image by changing the pixel size
            img_temp = cv2.resize(frame, (w // 20, h // 20), interpolation=cv2.INTER_LINEAR)
            blurred_img = cv2.resize(img_temp, (w, h), interpolation=cv2.INTER_NEAREST)

            # creating a mask by making a black background with a white circle on top of the face
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = cv2.circle(mask, getCenter(x,x1,y,y1), round(max([(x1-x),(y-y1)])/2 + 30), [255, 255, 255], -1)

            # applying mask where white is blurred image and black is normal image
            blur_mask = np.where(mask == np.array([255, 255, 255]), blurred_img, frame)



            #FILTER 2 - glasses and mustache
            overlap_filter = frame.copy()

            # function for processing filter images
            def process_image(image, reference_points, ajust_height):

                try:
                    # getting coordinates for specified reference points
                    M_x1, M_y1 = coords[reference_points[0]]
                    M_x2, M_y2 = coords[reference_points[1]]

                    # getting coordinates for middle point between reference points
                    x = round(M_x1 + ((M_x2 - M_x1)/2))
                    y = round(M_y1 + ((M_y2 - M_y1)/2))

                    # defining filter image width based on distance between reference points    
                    image_width = max([abs(M_x2 - M_x1),abs(M_y2 - M_y1)]) + ((M_x2 - M_x1)*0.7)

                    # getting filter image size and resizing
                    h, w = image.shape[:2]
                    image = cv2.resize(image, (round(image_width), round((min(image_width, w) / max(image_width, w)) * h)), interpolation=cv2.INTER_LINEAR)

                    # getting angle between reference points
                    angle = math.degrees(math.atan2(abs(M_y1-M_y2),abs(M_x1-M_x2)))
                    if (M_y1 > M_y2):
                        angle = -angle

                    # using the rotate image function using the calculated angle
                    image = rotate_image(image, angle)

                    # updating image size
                    h, w = image.shape[:2]
                    
                    # defining the offsets for the image starting point coordinates
                    x_offset = x - round(w/2)
                    y_offset = (y - round(w*0.35)) if ajust_height else (y - round(h/2))

                    # defining the endpoints of the image
                    x_end = x_offset + w
                    y_end = y_offset + h  


                    alpha_channel = image[:, :, 3] / 255 
                    overlay_colors = image[:, :, :3]

                    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

                    h, w = image.shape[:2]
                    background_subsection = overlap_filter[y_offset:y_end,x_offset:x_end]

                    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

                    overlap_filter[y_offset:y_end,x_offset:x_end] = composite 

                    return
                except:
                    return
                
            # adding the filter images
            mustache = cv2.imread("./img/mustache.png", cv2.IMREAD_UNCHANGED)
            glasses = cv2.imread("./img/glasses.png", cv2.IMREAD_UNCHANGED)

            # using the processing images function to add the filter images
            process_image(mustache, [48,54], True)
            process_image(glasses, [37,46], False)



            # FILTER 3 - gray lips
            custom_filter = frame.copy()

            # making the frame gray then converting back to BGR
            gray = cv2.cvtColor(custom_filter, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # creating a black background for the mask
            h, w = frame.shape[:2]
            mask = np.zeros((h,w,3), np.uint8)
            mask[:,:] = (0,0,0)

            # getting the coordinates for the lips and inside of mouth
            mouth = coords[48:60]
            mouthIn = coords[61:68]
            
            # drawing the lips white and the mouth inside as black
            cv2.fillPoly(mask, pts=[mouth], color=(255,255,255))
            cv2.fillPoly(mask, pts=[mouthIn], color=(0,0,0))

            # applying mask where white is gray image and black is colored image
            custom_filter = np.where(mask == np.array([255, 255, 255]), gray, custom_filter)



    else:
        blur_mask = frame
        overlap_filter = frame
        custom_filter = frame
        cv2.putText(frame, "NO FACE", (50,50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255))


    # Exibe resultado
    cv2.imshow(title, output)

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()
