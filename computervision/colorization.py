import cv2

# Read grayscale image
img = cv2.imread("grey.jpg", 0)

# Apply a color map
color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

# Show result
cv2.imshow("Original", img)
cv2.imshow("Colorized", color)

cv2.waitKey(0)
cv2.destroyAllWindows()
