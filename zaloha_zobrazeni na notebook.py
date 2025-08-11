#!/usr/bin/env python3
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO
import serial

# === Nastavení ===
DETECT_PERIOD   = 1.5
HSV_H_TOL       = 10
HSV_S_TOL       = 50
HSV_V_TOL       = 50
CONF_THRESH     = 0.4
IMG_W, IMG_H    = 640, 480
Kp, v_const     = 0.001, 0.6

# UART (odemkni dle potřeby)
ser = serial.Serial('/dev/serial0', 115200, timeout=1)

def compute_pwms(error):
    omega = Kp * error
    vl = max(0.0, min(1.0, v_const - omega))
    vr = max(0.0, min(1.0, v_const + omega))
    tl, tr = int(vl * 255), int(vr * 255)
    return tl, tr

# Sdílená data + zámky
data_lock    = threading.Lock()
frame_lock   = threading.Lock()
latest_frame = None
track_center = (IMG_W // 2, IMG_H // 2)
track_color  = (0, 0, 0)
have_color   = False
rectangle    = None

# Načti YOLO a kameru
model = YOLO('best.pt')
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

running = True

def detector_thread():
    global latest_frame, track_center, track_color, have_color, rectangle
    while running:
        time.sleep(DETECT_PERIOD)
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue

        H, W = frame.shape[:2]
        boxes = model(frame)[0].boxes
        bestd, best_c = float('inf'), None
        best_rect = None

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
                bestd = d
                best_c = (cx, cy)
                best_rect = (x1, y1, x2, y2)

        if best_c and best_rect:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx, cy = best_c
            r = 10
            x_lo, x_hi = max(0, cx - r), min(W, cx + r)
            y_lo, y_hi = max(0, cy - r), min(H, cy + r)
            roi = hsv[y_lo:y_hi, x_lo:x_hi]
            if roi.size == 0:
                continue
            hsv_mean = cv2.mean(roi)
            if hsv_mean is None:
                continue
            h, s, v = hsv_mean[:3]
            with data_lock:
                track_center = best_c
                track_color  = (h, s, v)
                have_color   = True
                rectangle    = best_rect

# Spusť detekční vlákno
det_thread = threading.Thread(target=detector_thread, daemon=True)
det_thread.start()

print("=== Hybridní tracker (s GUI výstupem) ===")
print("ESC zavře okno | Ctrl+C ukončí program\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # Ulož snímek pro detektor
        with frame_lock:
            latest_frame = frame.copy()

        H, W = frame.shape[:2]
        with data_lock:
            center = track_center
            color  = track_color
            valid  = have_color
            draw_rect = rectangle

        # HSV tracking
        if valid:
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = color
            h0 = int(h - HSV_H_TOL)
            h1 = int(h + HSV_H_TOL)
            s0 = max(0, min(255, int(s - HSV_S_TOL)))
            s1 = max(0, min(255, int(s + HSV_S_TOL)))
            v0 = max(0, min(255, int(v - HSV_V_TOL)))
            v1 = max(0, min(255, int(v + HSV_V_TOL)))

            if h0 < 0 or h1 > 179:
                # ČERVENÁ — rozdělit rozsah hue na 2 části
                lower1 = (0, s0, v0)
                upper1 = (h1 % 180, s1, v1)
                lower2 = (h0 % 180, s0, v0)
                upper2 = (179, s1, v1)
                mask1 = cv2.inRange(hsv_img, lower1, upper1)
                mask2 = cv2.inRange(hsv_img, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower = (h0, s0, v0)
                upper = (h1, s1, v1)
                mask = cv2.inRange(hsv_img, lower, upper)

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
                draw_rect = None

        # Spočti PWM
        if not valid:
            pwm_l, pwm_r = 0, 0
        else:
            err = (W / 2) - center[0]
            pwm_l, pwm_r = compute_pwms(err)

        # UART posílání
        ser.write(f"{pwm_l},{pwm_r}\n".encode())

        # Debug výpis
        print(f"[{time.strftime('%H:%M:%S')}] mode={'HSV' if valid else 'IDLE'}  "
              f"center={center}  PWM=({pwm_l},{pwm_r})  track_color={track_color}", flush=True)

        # === GUI BLOK ===
        if draw_rect:
            x1, y1, x2, y2 = draw_rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if valid and isinstance(center, tuple):
            cv2.circle(frame, center, 6, (0, 255, 255), -1)

        cv2.putText(frame, f"{'HSV' if valid else 'IDLE'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{pwm_l},{pwm_r}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Tracking Output", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        # === KONEC GUI BLOKU ===

        time.sleep(0.02)

except KeyboardInterrupt:
    running = False
    print("\n=== Ukončuji ===")

finally:
    det_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    print("Exiting…")
# pro zobrazeni na pripojenej monitor ---- export DISPLAY=:0 , python3 -u cube_best.py
# pro zobrazeni obrazu na notebooku ------ ssh -Y pi@10.42.0.1 echo $DISPLAY --- localhost:10.0
# , python3 -u cube_best.py


