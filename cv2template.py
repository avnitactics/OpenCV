import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    template = cv2.imread("template.jpeg")

    result = cv2.matchTemplate(
        frame,
        template,
        cv2.TM_CCOEFF_NORMED
    )

    threshold = 0.5

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > threshold: 
    
        h, w = template.shape[:2]

        top_left = max_loc

        bottom_right = (
            top_left[0] + w,
            top_left[1] + h
        )

        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            (0, 255, 0),
            2
        )

    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()