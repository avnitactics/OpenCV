import cv2 as cv

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read();

    cv.imshow('WebCam',frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break;
capture.release()
cv.destroyAllWindows()
