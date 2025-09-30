import cv2
import os
from datetime import datetime
import time

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open webcam
cap = cv2.VideoCapture(0)

# Create folder for snapshots
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Timer for automatic snapshots
last_snapshot_time = 0
snapshot_interval = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Region of interest (face only)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # Automatic snapshot every X seconds
        current_time = time.time()
        if current_time - last_snapshot_time >= snapshot_interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Auto snapshot saved: {filename}")
            last_snapshot_time = current_time

    # Show frame
    cv2.imshow("Face & Eye Detection", frame)

    # Key press options
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Manual snapshot saved: {filename}")

cap.release()
cv2.destroyAllWindows()
