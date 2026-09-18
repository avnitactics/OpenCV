import cv2
import numpy as np

print(f"Success! OpenCV version {cv2.__version__} loaded inside VS Code.")



img = cv2.imread("object_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,  # blockSize
    2)   # C

cv2.imshow("Adaptive", adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()