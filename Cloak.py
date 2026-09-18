import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(2)
ret, background = cap.read()

low = np.array([170, 120, 70])
high = np.array([180, 255, 255])

while ret:
	ret, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, low, high)
	inv_mask = cv2.bitwise_not(mask)
	
	cloak_area = cv2.bitwise_and(background, background, mask=mask)
	visible_area = cv2.bitwise_and(frame, frame, mask=inv_mask)
	final = cv2.add(cloak_area, visible_area)
	
	cv2.imshow("Color Mask", mask)
	cv2.imshow("Background Mask", inv_mask)
	cv2.imshow("Masked ROI", cloak_area)
	cv2.imshow("Background", visible_area)
	cv2.imshow("Invisibility Cloak", final)
	
	if cv2.waitKey(1) == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
