#!/usr/bin/env python3
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO
import serial

# === Nastavení ===
DETECT_PERIOD   = 1.5      # interval YOLO detekcí (s)
HSV_H_TOL       = 10       # tolerance Hue
HSV_S_TOL       = 50       # tolerance Sat.
HSV_V_TOL       = 50       # tolerance Val.
CONF_THRESH     = 0.4      # prahová důvěra YOLO
IMG_W, IMG_H    = 640, 480 # rozlišení kamery
Kp, v_const     = 0.0009, 0.7

# UART (odemkni dle potřeby)
ser = serial.Serial('/dev/serial0', 115200, timeout=1)

def compute_pwms(error):
    """P-regulátor + převod na 0–255."""
    omega = Kp * error
    vl = max(0.0, min(1.0, v_const - omega))
    vr = max(0.0, min(1.0, v_const + omega))
    return int(vl * 255), int(vr * 255)

# Sdílená data mezi vlákny
data_lock    = threading.Lock()
frame_lock   = threading.Lock()
latest_frame = None
track_center = (IMG_W // 2, IMG_H // 2)
track_color  = (0, 0, 0)
have_color   = False
rectangle    = None

# Načti YOLO a otevři kameru
model = YOLO('best.pt')
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

running = True

def detector_thread():
    """Vlákno, které co DETECT_PERIOD s provede YOLO detekci."""
    global latest_frame, track_center, track_color, have_color, rectangle
    while running:
        time.sleep(DETECT_PERIOD)
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue

        H, W = frame.shape[:2]
        # jednorázová inference
        boxes = model(frame)[0].boxes
        bestd, best_c, best_rect = float('inf'), None, None

        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            # filtrovat podle tvaru kostky
            if not (0.65 <= w/h <= 1.35 and min(w, h) >= 30):
                continue
            cx, cy = x1 + w//2, y1 + h//2
            d = (W/2 - cx)**2 + (H - cy)**2
            if d < bestd:
                bestd, best_c, best_rect = d, (cx, cy), (x1, y1, x2, y2)

        if best_c is not None:
            # spočti průměr HSV v okolí středu
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx, cy = best_c; r = 10
            x_lo, x_hi = max(0, cx - r), min(W, cx + r)
            y_lo, y_hi = max(0, cy - r), min(H, cy + r)
            roi = hsv[y_lo:y_hi, x_lo:x_hi]
            if roi.size == 0:
                continue
            h, s, v, _ = cv2.mean(roi)
            # aktualizuj sdílené proměnné
            with data_lock:
                track_center = best_c
                track_color  = (h, s, v)
                have_color   = True
                rectangle    = best_rect

# start detector vlákna
det_thread = threading.Thread(target=detector_thread, daemon=True)
det_thread.start()

print("=== Hybridní tracker (headless) ===")
print("Ctrl+C ukončí program\n")

try:
    while True:
        # přečti nový snímek
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # ulož pro detekci
        with frame_lock:
            latest_frame = frame.copy()

        # načti sdílené výsledky
        with data_lock:
            center    = track_center
            color     = track_color
            valid     = have_color
            rect      = rectangle

        # HSV-tracking, pokud už máme barvu
        if valid:
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = color
            # uprav hranice s ořezem 0–179/0–255
            h0, h1 = h - HSV_H_TOL, h + HSV_H_TOL
            s0, s1 = max(0, s-HSV_S_TOL), min(255, s+HSV_S_TOL)
            v0, v1 = max(0, v-HSV_V_TOL), min(255, v+HSV_V_TOL)

            # speciál pro červenou (hue kolem 0)
            if h0 < 0 or h1 > 179:
                lower1, upper1 = (0, s0, v0), (h1%180, s1, v1)
                lower2, upper2 = (h0%180, s0, v0), (179, s1, v1)
                m1 = cv2.inRange(hsv_img, lower1, upper1)
                m2 = cv2.inRange(hsv_img, lower2, upper2)
                mask = cv2.bitwise_or(m1, m2)
            else:
                mask = cv2.inRange(hsv_img, (h0, s0, v0), (h1, s1, v1))

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bestA, bestPt = 0, None
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if not (0.65 <= w/h <= 1.35 and min(w,h) >= 30):
                    continue
                A = cv2.contourArea(c)
                if A > bestA:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        bestA = A
                        bestPt = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            if bestPt and bestA > 50:
                center = bestPt
            else:
                valid = False
                rect  = None

        # spočítej PWM (nebo 0 pokud invalid)
        if not valid:
            pwm_l, pwm_r = 0, 0
        else:
            err = (IMG_W//2) - center[0]
            pwm_l, pwm_r = compute_pwms(err)

        # pošli po UART
        ser.write(f"{pwm_l},{pwm_r}\n".encode())

        # jediný výstup do terminalu
        print(f"[{time.strftime('%H:%M:%S')}] mode={'HSV' if valid else 'IDLE'}  "
              f"center={center}  PWM=({pwm_l},{pwm_r})  color={track_color}", flush=True)

        time.sleep(0.02)

except KeyboardInterrupt:
    running = False
    print("\n=== Ukončuji ===")

finally:
    det_thread.join()
    cap.release()
    ser.close()
    print("Exiting…")

#  git add .
#  1041  git commit -m "feat: přidání nových funkcí pro RGB detekci kostek"
#  1042  git checkout -b feature/rgb-detection
#  1043  git push -u origin feature/rgb-detection