import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)
ret, background = cap.read()

low_red1 = np.array([0, 120, 70])
high_red1 = np.array([7, 255, 255])
low_red2 = np.array([170, 120, 170])
high_red2 = np.array([180, 255, 255])
low_green = np.array([35, 50, 50])
high_green = np.array([85, 255, 255])
low_yellow = np.array([15, 100, 100])
high_yellow = np.array([35, 255, 255])
low_blue = np.array([100, 50, 50])
high_blue = np.array([130, 255, 255])
low_violet = np.array([130, 40, 40])
high_violet = np.array([160, 255, 255])
low_orange = np.array([0,150,150])
high_orange = np.array([25, 255, 255])


while ret:
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#mask1 = cv2.inRange(hsv, low_red1, high_red1)
	#mask2 = cv2.inRange(hsv, low_red2, high_red2)
	#mask = cv2.bitwise_or(mask1, mask2)
	mask = cv2.inRange(hsv, low_green, high_green)
	inv_mask = cv2.bitwise_not(mask)
	
	cloak_area = cv2.bitwise_and(background, background, mask=mask)
	visible_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
	final = cv2.add(cloak_area, visible_area)
	
	cv2.imshow("Color Mask", mask)
	#cv2.imshow("Background Mask", inv_mask)
	#cv2.imshow("Masked ROI", cloak_area)
	#cv2.imshow("Background", visible_area)
	cv2.imshow("Invisibility Cloak", final)
	
	if cv2.waitKey(1) == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
