import cv2
import os
import numpy as np
from PIL import Image


def train_recognizer(data_path, output_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def get_image_id(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        ids = []
        for image_path in image_paths:
            face_image = Image.open(image_path).convert('L')
            face_np = np.array(face_image)
            unique_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(face_np)
            ids.append(unique_id)
            # cv2.imshow("Training", face_np)
            cv2.waitKey(1)
        return ids, faces

    IDs, face_data = get_image_id(data_path)
    recognizer.train(face_data, np.array(IDs))
    recognizer.write(output_path)
    cv2.destroyAllWindows()
    print("Training Completed")

data_path = "datasets"
output_path = "Trainer.yml"
train_recognizer(data_path, output_path)
