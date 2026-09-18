import cv2 as cv

img = cv.imread('photos/cats.jpg')
cv.imshow('Cats', img)

# Averaging
# average = cv.blur(img, (3,3))
# cv.imshow('Average Blur', average)

# Gaussian Blur
# gauss = cv.GaussianBlur(img, (3,3), 0)
# cv.imshow('Gaussian Blur', gauss)

# Median Blur
# median = cv.medianBlur(img, 3)
# cv.imshow('Median Blur', median)

# Bilateral
#d determines the size of the neighborhood considered around each pixel.
# sigmacolor determines how much the colors in the neighborhood will influence the blurring.
#smaller sigmacolor means that only colors very similar to the central pixel will be considered, while larger values mean that a wider range of colors will be included.
# sigmaspace determines how much the spatial location influences the blurring.

bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)