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
objects_ids = []#list for saving cubes ids 
objects_centers=[]#list for saving cubes centers 
for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])#gets the class of cube (id) 
            #bounding boxes
            x1,y1,w,h = box.xyxy [0] #gets coordinates of each cube and width and height 
            x1,y1,w,h = int(x1),int(y1),int(w),int(h)#turn values into int
            #object center 
            center_x,center_y = x1+(w/2),y1+(h/2)#vypocet stredu objektu pro lepsi lokalizaci medveda 
            center_x,center_y = int(center_x-x1/2), int(center_y-y1/2)#prevede hodnoty na int aby se dali pouzit ve funkci ukazujici stred 
            center = center_x,center_y
            objects_centers.append(center)#writes object centers into list 
            objects_ids.append(cls)#writes objects ids into list 
print(objects_ids)
print(objects_centers)


