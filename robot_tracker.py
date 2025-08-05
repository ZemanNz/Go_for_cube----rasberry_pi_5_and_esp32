#!/usr/bin/env python3
import time
import numpy as np
import cv2
from ultralytics import YOLO
import serial

# === Nastavení ===
DETECT_INTERVAL   = 25    # každých N snímků spustit YOLO
INITIAL_YOLO_CNT  = 3     # na začátku spustit YOLO 3×
HSV_H_TOL         = 10
HSV_S_TOL         = 50
HSV_V_TOL         = 50
CONF_THRESH       = 0.4
IMG_W, IMG_H      = 640, 480

# PID proměnné
Kp = 0.001
v_const = 0.5
old_l, old_r = 0, 0

def compute_pwms(error):
    omega = Kp * error
    vl = max(0.0, min(1.0, v_const - omega))
    vr = max(0.0, min(1.0, v_const + omega))
    tl, tr = int(vl*255), int(vr*255)
    global old_l, old_r
    alpha = 0.1
    pl = int(old_l + alpha*(tl-old_l))
    pr = int(old_r + alpha*(tr-old_r))
    old_l, old_r = pl, pr
    return pl, pr

# --- Inicializace (sériovou linku odkomentuj podle potřeby) ---
ser = serial.Serial('/dev/serial0', 115200, timeout=1)
time.sleep(2)

# --- Načti YOLO model ---
model = YOLO('best.pt')

# --- Otevři kameru ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

frame_count    = 0
initial_yolo   = 0
have_color     = False
track_color    = None
track_center   = (IMG_W//2, IMG_H//2)

# Pro akumulaci FPS
sum_time   = 0.0
sum_count  = 0
avg_fps    = 0.0

color_names = {2: "RED", 1: "GREEN", 0: "BLUE"}

print("=== Headless Hybrid Tracker ===")
print("Press Ctrl+C to stop\n")

try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        H, W = frame.shape[:2]

        # Urči zdroj pozice
        if initial_yolo < INITIAL_YOLO_CNT:
            use_yolo = True
            initial_yolo += 1
        else:
            use_yolo = not have_color or frame_count % DETECT_INTERVAL == 0

        mode = "YOLO" if use_yolo else "HSV"
        det_color = None
        det_xy    = None

        if use_yolo:
            # YOLO detekce
            boxes = model(frame)[0].boxes
            bestd, best_id, best_c = float('inf'), None, None

            for box in boxes:
                conf = float(box.conf[0])
                if conf < CONF_THRESH:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                w,h = x2-x1, y2-y1
                ar = w/h if h else 0
                if not (0.65 <= ar <= 1.35 and min(w,h) >= 30):
                    continue
                cx, cy = x1 + w//2, y1 + h//2
                d = (W/2 - cx)**2 + (H - cy)**2
                if d < bestd:
                    bestd, best_c, best_id = d, (cx,cy), int(box.cls[0])

            if best_c:
                track_center = best_c
                det_color    = color_names.get(best_id, str(best_id))
                det_xy       = best_c
                have_color   = True

                # uložíme referenční HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cx,cy = best_c; r=10
                x_lo,x_hi = max(0,cx-r), min(W,cx+r)
                y_lo,y_hi = max(0,cy-r), min(H,cy+r)
                roi = hsv[y_lo:y_hi, x_lo:x_hi]
                h,s,v,_ = cv2.mean(roi)
                track_color = (h,s,v)
            else:
                have_color = False

        else:
            # HSV tracking
            time.sleep(0.05)
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h,s,v = track_color
            h0 = max(0,min(179,int(h-HSV_H_TOL)));  h1 = max(0,min(179,int(h+HSV_H_TOL)))
            s0 = max(0,min(255,int(s-HSV_S_TOL)));  s1 = max(0,min(255,int(s+HSV_S_TOL)))
            v0 = max(0,min(255,int(v-HSV_V_TOL)));  v1 = max(0,min(255,int(v+HSV_V_TOL)))
            mask = cv2.inRange(hsv_img, (h0,s0,v0), (h1,s1,v1))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_area, best_pt = 0, None
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                ar = w/h if h else 0
                if not (0.65 <= ar <= 1.35 and min(w,h)>=30):
                    continue
                A = cv2.contourArea(c)
                if A > best_area:
                    M = cv2.moments(c)
                    if M['m00']>0:
                        best_area = A
                        best_pt   = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            if best_pt and best_area>50:
                track_center = best_pt
                det_color    = "TRACK"
                det_xy       = best_pt
            else:
                have_color = False

        # PWM
        if det_color is None:
            pwm_l, pwm_r = 0, 0
        else:
            error = (W/2) - track_center[0]
            pwm_l, pwm_r = compute_pwms(error)
        ser.write(f"{pwm_l},{pwm_r}\n".encode())

        # měření loop time a akumulace
        loop_time = time.time() - t0
        sum_time  += loop_time
        sum_count += 1
        if sum_count >= 10:
            avg_fps   = sum_count / sum_time
            sum_time  = 0.0
            sum_count = 0

        # výpis
        print(f"[{mode}] center={track_center}  color={det_color}  "
              f"det_xy={det_xy}  PWM={pwm_l},{pwm_r}  FPS={avg_fps:.1f}")

        frame_count += 1

except KeyboardInterrupt:
    print("\n=== Stopped by user ===")

finally:
    cap.release()
    ser.close()
    print("Exiting...")
