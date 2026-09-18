import cv2

img = cv2.imread("neymar.jpg")

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(
    img,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imshow("original", img)
cv2.imshow("binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
