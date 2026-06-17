#!/usr/bin/env python3
"""
HSV Color Calibrator
=====================
Tool standalone (TIDAK perlu ROS) untuk mencari nilai HSV range dari objek
target (misal landing pad / marker) menggunakan webcam, video file, atau foto.

Cara pakai:
    # Dari webcam (default kamera index 0)
    python3 hsv_calibrator.py --source webcam --index 0

    # Dari video file
    python3 hsv_calibrator.py --source video --path test.mp4

    # Dari foto
    python3 hsv_calibrator.py --source image --path sample.jpg

Geser-geser trackbar (H, S, V min/max) sampai HANYA objek target yang
muncul putih di window "Mask". Tekan 's' untuk save nilai ke hsv_values.txt,
'q' untuk keluar.
"""

import cv2
import numpy as np
import argparse
import json
import os


def nothing(x):
    pass


def create_trackbars(window_name):
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, 400, 350)
    cv2.createTrackbar("H min", window_name, 8, 179, nothing)
    cv2.createTrackbar("H max", window_name, 28, 179, nothing)
    cv2.createTrackbar("S min", window_name, 80, 255, nothing)
    cv2.createTrackbar("S max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V min", window_name, 60, 255, nothing)
    cv2.createTrackbar("V max", window_name, 255, 255, nothing)


def get_trackbar_values(window_name):
    h_min = cv2.getTrackbarPos("H min", window_name)
    h_max = cv2.getTrackbarPos("H max", window_name)
    s_min = cv2.getTrackbarPos("S min", window_name)
    s_max = cv2.getTrackbarPos("S max", window_name)
    v_min = cv2.getTrackbarPos("V min", window_name)
    v_max = cv2.getTrackbarPos("V max", window_name)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def process_frame(frame, window_name, min_area=300):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = get_trackbar_values(window_name)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    # Kurangi noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Cari contour terbesar dan gambar bounding box + centroid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                display, f"area={int(area)} cx={cx} cy={cy}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    return display, mask, result, lower, upper


def main():
    parser = argparse.ArgumentParser(description="HSV Color Calibrator")
    parser.add_argument("--source", choices=["webcam", "video", "image"], default="webcam")
    parser.add_argument("--index", type=int, default=0, help="Webcam index")
    parser.add_argument("--path", type=str, default=None,
                         help="Path video/image (default: data/raw/)")
    parser.add_argument("--min-area", type=int, default=300, help="Minimum contour area to draw box")
    parser.add_argument("--output", type=str, default="models/hsv_values.json",
                         help="Path output file hasil kalibrasi")
    args = parser.parse_args()

    window_name = "HSV Calibration"
    create_trackbars(window_name)

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.index)
    elif args.source == "video":
        if not args.path:
            print("Error: --path wajib diisi untuk source video")
            return
        cap = cv2.VideoCapture(args.path)
    else:  # image
        if not args.path:
            print("Error: --path wajib diisi untuk source image")
            return
        cap = None
        static_frame = cv2.imread(args.path)
        if static_frame is None:
            print(f"Error: tidak bisa baca gambar di {args.path}")
            return

    print("Kontrol:")
    print("  's' = save nilai HSV ke hsv_values.txt")
    print("  'q' = keluar")

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                if args.source == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
                    continue
                print("Gagal membaca frame.")
                break
        else:
            frame = static_frame.copy()

        display, mask, result, lower, upper = process_frame(frame, window_name, args.min_area)

        cv2.imshow("Original (with detection)", display)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            data = {
                "lower_hsv": list(int(v) for v in lower),
                "upper_hsv": list(int(v) for v in upper),
            }
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Tersimpan ke {args.output}: {data}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()