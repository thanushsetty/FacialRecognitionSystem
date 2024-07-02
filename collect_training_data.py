import cv2
import time

def datacollect():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Failed to open webcam.")
        return

    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    name = input("Enter Your Name: ")
    unique_id = int(time.time())  # Using current timestamp as a unique ID
    count = 0

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f'datasets/{name}.{unique_id}.{count}.jpg', gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        if count > 200:
            break

    video.release()
    cv2.destroyAllWindows()

    print("Dataset Collection Done.")

datacollect()
