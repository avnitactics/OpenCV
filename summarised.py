import cv2
import numpy as np
# Read image
img = cv2.imread("document.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ------------------------------------------------
# 1. SIMPLE BINARY THRESHOLDING
# ------------------------------------------------

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)


# ------------------------------------------------
# 2. BINARY INVERSE
# ------------------------------------------------

_, binary_inv = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY_INV
)











# ------------------------------------------------
# 6. OTSU'S THRESHOLDING
# ------------------------------------------------

_, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ------------------------------------------------
# 7. ADAPTIVE MEAN THRESHOLDING
# ------------------------------------------------

adaptive_mean = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)


# ------------------------------------------------
# 8. ADAPTIVE GAUSSIAN THRESHOLDING
# ------------------------------------------------

adaptive_gaussian = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)


# ------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)

cv2.imshow("Binary", binary)




cv2.imshow("Otsu", otsu)

cv2.imshow("Adaptive Mean", adaptive_mean)
cv2.imshow("Adaptive Gaussian", adaptive_gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()