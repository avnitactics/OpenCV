import cv2

img = cv2.imread("neymar.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 100, 200)

#edges2 = cv2.Canny(gray, 100, 200, 3, 5)
#edgesblur = cv2.Canny(blurred, 100, 200)

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
#cv2.imshow("Edges2", edges2)
#cv2.imshow("Edges (Blur)", edgesblur)

#sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#abs_sobel_x = cv2.convertScaleAbs(sobel_x)
#abs_sobel_y = cv2.convertScaleAbs(sobel_y)
#combined_edges = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
#cv2.imshow("x", abs_sobel_x)
#cv2.imshow("y", abs_sobel_y)
#cv2.imshow("combined", combined_edges)


cv2.waitKey(0)
cv2.destroyAllWindows()
