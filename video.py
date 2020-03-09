import cv2
import random

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = 128
    # gray_binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

    # Detect the faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        crop_img = img[y:y + h, x:x + w]
        crop_img = cv2.bitwise_not(crop_img)
        # thresh = 128
        # crop_img = cv2.threshold(crop_img, thresh, 255, cv2.THRESH_BINARY)[1]
        # # cv2.imshow("cropped", crop_img)
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        for n in range(1, 3):
            x_offset = x + random.randint(-10, 10)
            y_offset = y + random.randint(-30, 30)
            f_x = float(random.randint(5, 15)/10)
            f_y = float(random.randint(5, 15) / 10)
            resized = cv2.resize(crop_img,None,fx=f_x,fy=f_y)
            try:
                img[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
            except ValueError:
                print('out of bound again')


        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()