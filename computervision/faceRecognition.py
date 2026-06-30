import cv2
import os
import numpy as np

faces = []
labels = []

folder = "faces"

for file in os.listdir(folder):

    img = cv2.imread(
        folder + "/" + file,
        0
    )

    faces.append(img)
    labels.append(0)


recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.save("trainer.yml")

print("Training completed")
