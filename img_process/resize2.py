import cv2 as cv

img = cv.imread('photos/large.jpg')
cv.imshow('Original Image', img)

# Downscale (decrease image size) with maintaining aspect ratio
scale = 30

width = int(img.shape[1] * scale / 100)
height = int(img.shape[0] * scale / 100)

dim = (width, height)

res1 = cv.resize(img, dim, interpolation=cv.INTER_AREA)

cv.imshow('Resized1', res1)

cv.waitKey(0)
cv.destroyAllWindows()