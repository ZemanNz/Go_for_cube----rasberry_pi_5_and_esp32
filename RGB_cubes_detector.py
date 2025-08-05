#this program is for actual use on robot
 
#imports
from ultralytics import YOLO
import numpy as np
import cv2 
#import cvzone
import time
import matplotlib.pyplot as plt
import serial


ser = serial.Serial('/dev/serial0', 115200, timeout=1)
time.sleep(2)  # nech interface nastartovat
#loading model 
model = YOLO('best.pt')  # load a custom model
#capturing image
cap = cv2.VideoCapture(0)#inicialize camera 
cap.set(3,320)#img width
cap.set(4,240)#img height

Kp = 0.001
v_const = 0.5   # základní rychlost vpřed (0..1)

def compute_pwms(error):
    # P-only korekce na differential drive:
    omega = Kp * error
    # Levé/pravé relativní rychlosti
    vl = v_const - omega
    vr = v_const + omega
    # Oříznout do [0,1] (žádný záporný režim)
    vl = max(0.0, min(1.0, vl))
    vr = max(0.0, min(1.0, vr))
    # Převedení na 0–255
    return int(vl * 255), int(vr * 255)

def get_horizontal_error(center_x):
    # Vypočítá horizontální chybu na základě středu detekované kostky
    output = img_width/2 - center_x  # předpokládá se, že střed obrazu je v x=320
    print(f"Horizontal error: {output}")
    return output
#///////////////////////////////////

# odeslat zprávu
ser.write(b'Ahoj ESP32!\n')
print("Odesláno: Ahoj ESP32!")

old_pwm_l = 128
old_pwm_r = 128

while True:
    sucess, img = cap.read()#reads frame from camera 
    if not sucess:
        
        print("Camera read failed.")
        break

    results = model(img)#applies model on image
    objects_ids = []
    objects_centers = []
    cube_distances = []
    img_height, img_width = img.shape[:2]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cube_id = int(box.cls[0])  # gets the class of cube (id)
            # map id to color name
            color_names = {2: "RED", 1: "GREEN", 0: "BLUE"}
            color_name = color_names.get(cube_id, str(cube_id))
            #bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1

            # validace kostky podle poměru stran a velikosti
            aspect_ratio = width / height if height != 0 else 0
            min_side = min(width, height)
            if not (0.65 <= aspect_ratio <= 1.35 and min_side >= 30):
                continue  # přeskočí nevalidní "kostku"

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # správné vykreslení
            center_x, center_y = x1 + width // 2, y1 + height // 2
            center = center_x, center_y
            objects_centers.append(center)
            objects_ids.append(cube_id)
            # vykreslení středu a ID na obrázek
            # cv2.circle(img, center, 5, (0,255,0), -1)
            # cv2.putText(img, color_name, (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # vzdálenost od středu obrazu
            konstanta1 = abs((img_width/2)-center_x)
            cube_distance = ((konstanta1*konstanta1 + (img_height-center_y)*(img_height-center_y)))*0.5
            cube_distances.append(cube_distance)

    print("Detekované kostky:")
    color_names = {2: "RED", 1: "GREEN", 0: "BLUE"}
    for cube_id, center, dist in zip(objects_ids, objects_centers, cube_distances):
        color_name = color_names.get(cube_id, str(cube_id))
        print(f"  barva={color_name}, pozice={center}, vzdálenost={dist:.1f}")

    # najdi index nejbližší kostky
    if cube_distances:
        a = cube_distances.index(min(cube_distances))
        x, y = objects_centers[a]
        color_name = color_names.get(objects_ids[a], str(objects_ids[a]))
        print(f"Nejbližší kostka je {color_name} na pozici ({x},{y}), vzdálenost={cube_distances[a]:.1f}")
        # cv2.circle(img, (x, y), 5, (255,255,255), thickness=-1)
        error = get_horizontal_error(x)
    else:
        error = 0

    pwm_l, pwm_r = compute_pwms(error)
    smoothing = 0.1  # menší váha = pomalejší změna
    pwm_l = int(old_pwm_l + (pwm_l - old_pwm_l) * smoothing)
    pwm_r = int(old_pwm_r + (pwm_r - old_pwm_r) * smoothing)
    old_pwm_l, old_pwm_r = pwm_l, pwm_r

    msg = f"{pwm_l},{pwm_r}\n"
    ser.write(msg.encode())
    
    # Odstraň nebo zkrať prodlevu pro vyšší FPS
    # time.sleep(0.5)  # zpomalí smyčku na cca 2 FPS
    #time.sleep(0.01)    # cca 8-10 FPS (dle výkonu)

cap.release()
cv2.destroyAllWindows()


