# Firslt we create a databases of Images from different peoples.
## step 1 ------


#Run on Terminal 1.->  & "C:\Users\Rahul Kumar Jha\anaconda3\shell\condabin\conda-hook.ps1"
#  2. conda activate dl 
# 3. cd Project_face_Recognition
#  4.python datasetcreates.py

import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
DATASET_PATH  = "dataset"
FACE_SIZE     = (100, 100)
AUTO_CAPTURE  = False
# ─────────────────────────────────────────────

def get_name_from_gui():
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    name = simpledialog.askstring(
        title="Face Dataset",
        prompt="Apna naam enter karo:",
        parent=root
    )
    root.destroy()
    if name:
        name = name.strip().title()
    return name


def main():
    person_name = get_name_from_gui()
    if not person_name:
        print("Naam nahi diya. Exiting.")
        return

    print(f"\nCapturing dataset for: {person_name}")

    person_path = os.path.join(DATASET_PATH, person_name)
    os.makedirs(person_path, exist_ok=True)

    existing = [f for f in os.listdir(person_path) if f.lower().endswith(".jpg")]
    count = len(existing)
    print(f"Pehle se {count} images hain.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam nahi mili!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    auto_counter  = 0
    STABLE_FRAMES = 8

    print("Instructions:")
    if AUTO_CAPTURE:
        print("  → Face saamne rakho — automatically capture hoga")
    else:
        print("  → SPACE dabao capture ke liye")
    print("  → Q dabao jab enough images aa jayein\n")  # ✅ updated message

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq      = cv2.equalizeHist(gray)
        gray_blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        faces = face_cascade.detectMultiScale(
            gray_blurred,
            scaleFactor  = 1.05,
            minNeighbors = 6,
            minSize      = (80, 80),
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        face_found = len(faces) > 0
        display    = frame.copy()

        # ✅ Sirf count dikhao — koi progress bar nahi, koi limit nahi
        cv2.putText(display, f"{person_name}  [Captured: {count}]",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(display, "SPACE = Capture  |  Q = Quit & Save",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if face_found:
            (x, y, w, h) = faces[0]

            if AUTO_CAPTURE:
                auto_counter += 1
                bar_fill = int((auto_counter / STABLE_FRAMES) * w)
                cv2.rectangle(display, (x, y + h + 4),
                              (x + bar_fill, y + h + 12), (0, 255, 0), -1)
                cv2.rectangle(display, (x, y + h + 4),
                              (x + w,       y + h + 12), (0, 255, 0), 1)
            else:
                auto_counter = 0

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # AUTO CAPTURE
            should_capture = AUTO_CAPTURE and auto_counter >= STABLE_FRAMES
            if should_capture:
                auto_counter  = 0
                face_color    = frame[y:y + h, x:x + w]
                face_resized  = cv2.resize(face_color, FACE_SIZE,
                                           interpolation=cv2.INTER_CUBIC)
                count        += 1
                save_path     = os.path.join(person_path, f"{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                print(f"  Saved {count} — {save_path}")

        else:
            auto_counter = 0
            cx, cy, bw, bh = 320, 240, 200, 200
            cv2.rectangle(display,
                          (cx - bw // 2, cy - bh // 2),
                          (cx + bw // 2, cy + bh // 2),
                          (0, 0, 255), 2)
            cv2.putText(display, "Face saamne laao",
                        (cx - 95, cy + bh // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Dataset Capture", display)
        key = cv2.waitKey(1) & 0xFF

        # ✅ SPACE — unlimited capture, sirf face_found check
        if key == ord(' ') and not AUTO_CAPTURE:
            if face_found:
                (x, y, w, h) = faces[0]
                face_color    = frame[y:y + h, x:x + w]
                face_resized  = cv2.resize(face_color, FACE_SIZE,
                                           interpolation=cv2.INTER_CUBIC)
                count        += 1
                save_path     = os.path.join(person_path, f"{count}.jpg")
                cv2.imwrite(save_path, face_resized)
                print(f"  Saved {count}")
            else:
                print("  Koi face detect nahi hua.")

        elif key == ord('q'):
            break   # ✅ Q dabao jab chahein — 10 ho ya 200

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Total {count} images saved for '{person_name}' → {person_path}")
    print("Ab face_train.py chalao.")


if __name__ == "__main__":
    main()