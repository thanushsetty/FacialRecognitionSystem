import cv2
import os
import numpy as np
from PIL import Image


def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    name_dict = {}

    def get_image_info(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        for image_path in image_paths:
            face_image = Image.open(image_path).convert('L')
            face_np = np.array(face_image)
            unique_id = int(os.path.split(image_path)[-1].split(".")[1])
            name = os.path.split(image_path)[-1].split(".")[0]
            name_dict[unique_id] = name
            faces.append(face_np)
            ids.append(unique_id)
            cv2.waitKey(1)
        return ids, faces

    IDs, face_data = get_image_info("datasets")
    recognizer.train(face_data, np.array(IDs))

    blink_counter = 0
    prev_eye_state = True
    liveliness_detected = False
    detected_name = ""
    detected_blink_count = 0

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Failed to open webcam.")
        return

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            gray_face = gray[y:y+h, x:x+w]
            serial, conf = recognizer.predict(gray_face)

            if conf <= 50:
                detected_name = name_dict.get(serial, "Unknown")
            else:
                detected_name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, detected_name, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(gray_face)
            mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            current_eye_state = len(eyes) > 0

            if prev_eye_state and not current_eye_state:
                blink_counter += 1
            prev_eye_state = current_eye_state

            if len(eyes) == 0:
                cv2.putText(frame, 'Eyes Close', (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye Open", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                mouth_roi = roi_gray[my:my+mh, mx:mx+mw]
                white_pixels = np.count_nonzero(mouth_roi > 200)
                if white_pixels > 10:
                    cv2.putText(roi_color, "Mouth Open", (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Smile Detected", (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if blink_counter >= 3:
                        liveliness_detected = True
                else:
                    cv2.putText(roi_color, "Mouth Closed", (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if liveliness_detected:
                cv2.putText(frame, "Liveliness Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)

        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k == ord("q"):
            break

    video.release()


face_recognition()
cv2.destroyAllWindows()