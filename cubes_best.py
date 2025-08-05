#!/usr/bin/env python3
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO
import serial

# === Nastavení ===
DETECT_PERIOD   = 0.5    # detektor YOLO každých 0.5 s
HSV_H_TOL       = 10
HSV_S_TOL       = 50
HSV_V_TOL       = 50
CONF_THRESH     = 0.4
IMG_W, IMG_H    = 640, 480

Kp, v_const     = 0.001, 0.5
old_l, old_r    = 0, 0

# UART (odemkni, až budeš posílat na ESP32)
# ser = serial.Serial('/dev/serial0', 115200, timeout=1)

def compute_pwms(error):
    omega = Kp * error
    vl = max(0.0, min(1.0, v_const - omega))
    vr = max(0.0, min(1.0, v_const + omega))
    tl, tr = int(vl * 255), int(vr * 255)
    global old_l, old_r
    alpha = 0.1
    pl = int(old_l + alpha * (tl - old_l))
    pr = int(old_r + alpha * (tr - old_r))
    old_l, old_r = pl, pr
    return pl, pr

# Sdílená data + zámky
data_lock    = threading.Lock()
frame_lock   = threading.Lock()
latest_frame = None
track_center = (IMG_W // 2, IMG_H // 2)
track_color  = (0, 0, 0)
have_color   = False
rectangle   = 0, 0, 0, 0

# Načti YOLO a kameru
model = YOLO('best.pt')
cap   = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

running = True

def detector_thread():
    global latest_frame, track_center, track_color, have_color, rectangle
    while running:
        time.sleep(DETECT_PERIOD)
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            continue

        H, W = frame.shape[:2]
        boxes = model(frame)[0].boxes
        bestd, best_c, best_id = float('inf'), None, None
        best_x, best_y, best_w, best_h = 0, 0, 0, 0
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            ar = w / h if h else 0
            if not (0.65 <= ar <= 1.35 and min(w, h) >= 30):
                continue
            cx, cy = x1 + w // 2, y1 + h // 2
            d = (W/2 - cx)**2 + (H - cy)**2
            if d < bestd:
                bestd, best_c, best_id = d, (cx, cy), int(box.cls[0])
                best_x, best_y, best_w, best_h = x1, y1, w, h

        if best_c:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx, cy = best_c
            r = 10
            x_lo, x_hi = max(0, cx - r), min(W, cx + r)
            y_lo, y_hi = max(0, cy - r), min(H, cy + r)
            roi = hsv[y_lo:y_hi, x_lo:x_hi]
            h, s, v, _ = cv2.mean(roi)
            rectangle = best_x, best_y, best_w, best_h
            with data_lock:
                track_center = best_c
                track_color  = (h, s, v)
                have_color   = True

# Spusť detekční vlákno
det_thread = threading.Thread(target=detector_thread, daemon=True)
det_thread.start()

print("=== Hybridní tracker (debug, GUI + UART je odkomentovaný) ===")
print("Esc zavře okno a Ctrl+C ukončí program\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # Ulož aktuální snímek pro detektor
        with frame_lock:
            latest_frame = frame.copy()

        H, W = frame.shape[:2]
        with data_lock:
            center = track_center
            color  = track_color
            valid  = have_color

        mode = "HSV" if valid else "IDLE"

        if valid:
            # HSV tracking
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = color
            h0 = max(0,   min(179, int(h - HSV_H_TOL)))
            h1 = max(0,   min(179, int(h + HSV_H_TOL)))
            s0 = max(0,   min(255, int(s - HSV_S_TOL)))
            s1 = max(0,   min(255, int(s + HSV_S_TOL)))
            v0 = max(0,   min(255, int(v - HSV_V_TOL)))
            v1 = max(0,   min(255, int(v + HSV_V_TOL)))
            mask = cv2.inRange(hsv_img, (h0, s0, v0), (h1, s1, v1))
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_area, best_pt = 0, None
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                ar = w / h if h else 0
                if not (0.65 <= ar <= 1.35 and min(w, h) >= 30):
                    continue
                A = cv2.contourArea(c)
                if A > best_area:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        best_area = A
                        best_pt   = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            if best_pt and best_area > 50:
                center = best_pt
            else:
                valid = False
                mode  = "IDLE"

        # Spočti PWM
        if not valid:
            pwm_l, pwm_r = 0, 0
        else:
            err = (W/2) - center[0]
            pwm_l, pwm_r = compute_pwms(err)

        # UART (odemkni, chceš-li posílat)
        # ser.write(f"{pwm_l},{pwm_r}\n".encode())

        # Vykresli a zobraz
        a,b,c,d = rectangle
        cv2.rectangle(frame, (a, b), (a + c, b + d), (0, 255, 255), 2)
        cv2.circle(frame, center, 6, (0,255,255), -1)
        cv2.putText(frame, mode, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"PWM {pwm_l},{pwm_r}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Hybrid Tracker Debug", frame)

        # Výpis do terminálu
        print(f"[{mode}] center={center}  PWM={pwm_l},{pwm_r}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        time.sleep(0.02)
finally:
    running = False
    det_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting…")
