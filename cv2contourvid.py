import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(
    frame,
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

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 5000:
            continue
        perimeter = cv2.arcLength(contour, True)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Calculate center of contour
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Display area
            cv2.putText(
                frame,
                f"Area: {int(area)}",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # Display perimeter
            cv2.putText(
                frame,
                f"Peri: {int(perimeter)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

            # Display contour center
            cv2.circle(
                frame,
                (cX, cY),
                5,
                (0, 0, 255),
                -1
            )

    # cv2.drawContours(
    #     frame,
    #     contours,
    #     -1,
    #     (0, 255, 0),
    #     2
    # )

    cv2.imshow("binary", binary)
    cv2.imshow("frame", frame)
cap.release()
cv2.destroyAllWindows()

#img = cv2.imread("neymar.jpg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#edges = cv2.Canny(gray, 100, 200)

#cv2.imshow("Original", img)
#cv2.imshow("Edges", edges)

#cv2.waitKey(0)
#cv2.destroyAllWindows()