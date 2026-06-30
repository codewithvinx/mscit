import cv2
import numpy as np

img1 = cv2.imread("car1.jpg", 0)
img2 = cv2.imread("car2.jpg", 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matches = cv2.BFMatcher(cv2.NORM_HAMMING).match(des1, des2)

p1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
p2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                         matchesMask=mask.ravel().tolist())

cv2.imshow("RANSAC", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
