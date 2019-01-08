import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)

while(True): # gay
    ref, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        print(x, y, w, h)
        roi_grey = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my-image2.png"
        cv2.imwrite(img_item, roi_grey)

        color = (0, 0, 255)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()