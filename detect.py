from ultralytics import YOLO
import cv2


model = YOLO('yolov8s.pt')


cap = cv2.VideoCapture('video.mp4')  # Use 0 for the default camera
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break


    results = model(frame)
    annoted_frame = results[0].plot()

    cv2.imshow('YOLOv8 Detetion',annoted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()