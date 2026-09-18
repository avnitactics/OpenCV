import cv2 as cv

capture = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    isTrue, frame = capture.read();

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        
    cv.imshow('WebCam',frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break;
capture.release()
cv.destroyAllWindows()