# face_recognition.py
# Step 3 — Live face pehchano aur naam dikhao
# Run karo: python face_recognition.py

import cv2
import json
import numpy as np
from collections import deque   # ✅ NEW — flickering fix ke liye

FACE_SIZE            = (100, 100)
CONFIDENCE_THRESHOLD = 110        # ✅ CHANGED: 90 → 80 (90 bahut strict tha)
SMOOTH_FRAMES        = 3         # ✅ NEW — last 7 frames ka average lo

def load_resources():
    try:
        with open("labels.json", "r") as f:
            raw = json.load(f)
        label_map = {int(k): v for k, v in raw.items()}
        print(f"✓ Labels loaded: {label_map}")
    except FileNotFoundError:
        print("ERROR: labels.json nahi mili!")
        return None, None

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read("face_model.yml")
        print("✓ Model loaded: face_model.yml")
    except Exception as e:
        print(f"ERROR: face_model.yml load nahi hua — {e}")
        return None, None

    return face_recognizer, label_map


def main():
    print("=" * 40)
    print("  Face Recognition")
    print("=" * 40)

    face_recognizer, label_map = load_resources()
    if face_recognizer is None:
        return

    # ✅ CHANGED — 2 cascades load karo (frontal + alt2 for tilt)
    cascade_main = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cascade_alt = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam nahi mili!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    # ✅ NEW — Smoothing buffer (flickering fix)
    label_buffer = deque(maxlen=SMOOTH_FRAMES)
    conf_buffer  = deque(maxlen=SMOOTH_FRAMES)

    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq   = cv2.equalizeHist(gray)
        gray_blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        # ✅ CHANGED — minNeighbors: 6→4, minSize: (80,80)→(40,40)
        # minNeighbors kam = tilt me bhi detect karega
        # minSize kam = door se bhi detect karega
        faces = cascade_main.detectMultiScale(
            gray_blur,
            scaleFactor  = 1.05,
            minNeighbors = 4,           # ✅ CHANGED: 6 → 4
            minSize      = (40, 40),    # ✅ CHANGED: (80,80) → (40,40)
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        # ✅ NEW — Agar main cascade ne detect nahi kiya toh alt2 try karo
        if len(faces) == 0:
            faces = cascade_alt.detectMultiScale(
                gray_blur,
                scaleFactor  = 1.05,
                minNeighbors = 4,
                minSize      = (40, 40),
                flags        = cv2.CASCADE_SCALE_IMAGE
            )

        for (x, y, w, h) in faces:
            face_crop    = gray[y:y + h, x:x + w]
            face_eq      = cv2.equalizeHist(face_crop)
            face_resized = cv2.resize(face_eq, FACE_SIZE,
                                      interpolation=cv2.INTER_CUBIC)

            label, confidence = face_recognizer.predict(face_resized)

            # ✅ NEW — Buffer mein dalo (smoothing)
            label_buffer.append(label)
            conf_buffer.append(confidence)

            # ✅ NEW — Majority vote lo last 7 frames se
            if len(label_buffer) == SMOOTH_FRAMES:
                # Sabse zyada aane wala label winner hai
                smooth_label = max(set(label_buffer),
                                   key=list(label_buffer).count)
                smooth_conf  = sum(conf_buffer) / len(conf_buffer)
            else:
                smooth_label = label
                smooth_conf  = confidence

            # ✅ CHANGED — Smoothed values se decision lo
            if smooth_conf < CONFIDENCE_THRESHOLD:
                name  = label_map.get(smooth_label, "Unknown")
                color = (0, 220, 0)
            else:
                name  = "Unknown"
                color = (0, 0, 220)

            bar_max  = 120
            bar_fill = max(0, int((1 - smooth_conf / bar_max) * w))
            cv2.rectangle(frame, (x, y + h + 4),
                          (x + bar_fill, y + h + 14), color, -1)
            cv2.rectangle(frame, (x, y + h + 4),
                          (x + w, y + h + 14), (180, 180, 180), 1)

            label_text = f"{name}  ({smooth_conf:.1f})"
            cv2.putText(frame, label_text, (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(frame, "Press Q to quit", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recognition band ho gayi.")


if __name__ == "__main__":
    main()
