import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("doggo.jfif")
img2 = cv2.imread("cat.jpg")

# Resize second image to same height
h = min(img1.shape[0], img2.shape[0])

img1 = cv2.resize(img1, (int(img1.shape[1]*h/img1.shape[0]), h))
img2 = cv2.resize(img2, (int(img2.shape[1]*h/img2.shape[0]), h))

# Join images horizontally
combined = cv2.hconcat([img1, img2])

# Display
combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

plt.imshow(combined)
plt.title("Combined Image")
plt.axis("off")
plt.show()
