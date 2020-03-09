import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('test2.jpg')
img_blur = cv2.bilateralFilter(img, d = 7,
                               sigmaSpace = 75, sigmaColor =75)
gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)


# Draw rectangle around the faces
for (x, y, w, h) in faces:
    face_area = cv2.rectangle(img, (x, y-130), (x+w, y+h+30), (255, 0, 0), 2)




rev = cv2.bitwise_not(gray)

thresh = 80
ret,thresh_img = cv2.threshold(rev, thresh, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Found " + str(len(contours)) + " counters")


img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(gray, contours, -1, (0,255,0), 2)
cv2.drawContours(img_contours, contours, -1, (0,255,0), 2)





#
cv2.imshow('gray', gray)
cv2.imshow('img', img_contours)

cv2.waitKey()
cv2.destroyAllWindows()


