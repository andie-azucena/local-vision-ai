from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
# model = YOLO("yolov8n.pt")
model = YOLO("yolov8s.pt")  # more accurate but slower

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO on each frame
    annotated_frame = results[0].plot()  # Draw boxes and labels

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
