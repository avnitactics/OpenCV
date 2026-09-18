import cv2
import numpy as np

print(f"Success! OpenCV version {cv2.__version__} loaded inside VS Code.")



img = cv2.imread("object_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



_, otsu = cv2.threshold(
    gray,
    0, 255,
    cv2.THRESH_BINARY +
    cv2.THRESH_OTSU)  

cv2.imshow("Otsu", otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()