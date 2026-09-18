import cv2
import numpy as np

print(f"Success! OpenCV version {cv2.__version__} loaded inside VS Code.")



img = cv2.imread("object_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, binary = cv2.threshold(
    gray, 127, 255,
    cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (3, 3))

eroded = cv2.erode(
    binary, kernel, iterations=1)

cv2.imshow("Eroded", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()