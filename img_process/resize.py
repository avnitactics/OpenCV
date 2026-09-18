import cv2 as cv

img =cv.imread('photos/cats.jpg')

cv.imshow('Original Image',img)

# resized=cv.resize(img,(500,500))
# cv.imshow('Resized Image',resized)

croped=img[100:500,100:500]
cv.imshow('Croped Image',croped)
cv.waitKey(0)