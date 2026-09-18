import cv2
import numpy as np

img = cv2.imread("document.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Kernel
kernel = np.ones((3, 3), np.uint8)

# Erosion
eroded = cv2.erode(
    binary, kernel, iterations=1
)

# Dilation
dilated = cv2.dilate(
    binary, kernel, iterations=1
)

cv2.imshow("Binary", binary)
cv2.imshow("Eroded", eroded)
cv2.imshow("Dilated", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()