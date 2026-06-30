import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("doggo.jfif")

# Convert BGR to RGB for displaying
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

# 1. Translation (shift image)
tx, ty = 100, 50
translation_matrix = np.float32([[1, 0, tx],
                                 [0, 1, ty]])
translated = cv2.warpAffine(img, translation_matrix, (w, h))

# 2. Rotation
angle = 45
center = (w//2, h//2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated = cv2.warpAffine(img, rotation_matrix, (w, h))


# 3. Scaling
scaled = cv2.resize(img, None, fx=1.5, fy=1.5)

# 4. Reflection (flip)
reflection = cv2.flip(img, 1)   # 1 = horizontal, 0 = vertical

# Display results
titles = [
    "Original",
    "Translated",
    "Rotated",
    "Scaled",
    "Reflected"
]

images = [
    img,
    translated,
    rotated,
    scaled,
    reflection
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.show()
