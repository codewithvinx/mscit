from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

img = cv2.imread("objectImage.jpeg")

results = model(img)

result_img = results[0].plot()

cv2.imshow(
    "Object Detection",
    result_img
)

cv2.waitKey(0)
cv2.destroyAllWindows()
