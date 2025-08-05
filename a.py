#imports
from ultralytics import YOLO
import numpy as np
import cv2 
#import cvzone
import time
import matplotlib.pyplot as plt
import serial

model = YOLO('best.pt')  # load a custom model 
print(model.info())
# Přibližná velikost modelu podle počtu parametrů:
params = 25858057  # získáno z výpisu info()
if params < 7_000_000:
    print("Detekovaný model: YOLOv8n (nano)")
elif params < 25_000_000:
    print("Detekovaný model: YOLOv8s (small)")
elif params < 70_000_000:
    print("Detekovaný model: YOLOv8m (medium)")
elif params < 120_000_000:
    print("Detekovaný model: YOLOv8l (large)")
else:
    print("Detekovaný model: YOLOv8x (extra large)")