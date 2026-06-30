import cv2
import pytesseract

# Set the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Read the image
img = cv2.imread("text.jpg")

# Extract text
text = pytesseract.image_to_string(img)

print(text)
