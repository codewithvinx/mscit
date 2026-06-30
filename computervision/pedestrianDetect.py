import cv2

# Load image
img = cv2.imread("street.jpg")

# Check image
if img is None:
    print("Image not found!")
    exit()


# Create HOG detector
hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)


# Detect pedestrians
people, weights = hog.detectMultiScale(
    img,
    winStride=(8,8),
    padding=(8,8),
    scale=1.05
)


# Draw boxes
for x, y, w, h in people:
    cv2.rectangle(
        img,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )


cv2.imshow(
    "Pedestrian Detection",
    img
)

cv2.waitKey(0)
cv2.destroyAllWindows()
