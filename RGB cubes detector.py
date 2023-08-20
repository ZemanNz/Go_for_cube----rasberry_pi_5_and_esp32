#this program is for actual use on robot
 
#imports
from ultralytics import YOLO
import numpy as np
import cv2 
import cvzone
import time
import matplotlib.pyplot as plt

#loading model 
model = YOLO('best.pt')  # load a custom model
#capturing image
cap = cv2.VideoCapture(0)#inicialize camera 
cap.set(3,640)#img width
cap.set(4,480)#img height
sucess, img = cap.read()#reads frame from camera 
#appliing model on image 
results = model(img)#applies model on image
