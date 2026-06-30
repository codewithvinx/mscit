import cv2
import numpy as np
from PIL import Image

img = cv2.imread("input.jpg")
mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)

rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = mask

fg = Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

bg = Image.open("background.jpg").convert("RGBA").resize(fg.size)

Image.alpha_composite(bg, fg).save("output.png")

print("Done")
