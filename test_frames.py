import cv2
cap = cv2.VideoCapture('songs/macarena.mp4')
c = 0
while True:
    ret, _ = cap.read()
    if not ret:
        break
    c += 1
print("Actual frames read:", c)
