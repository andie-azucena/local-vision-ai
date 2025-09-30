import cv2
from datetime import datetime
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# -------------------------------
# Initialize models
# -------------------------------

yolo_model = YOLO("yolov8n.pt")  # nano model for speed
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model.to(device)

# -------------------------------
# Setup webcam and snapshot folder
# -------------------------------

cap = cv2.VideoCapture(0)
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Variable to store latest caption
latest_caption = ""

# -------------------------------
# Main loop
# -------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = yolo_model(frame)
    annotated_frame = results[0].plot()  # Draw bounding boxes

    # Draw the latest caption on every frame
    if latest_caption:
        cv2.putText(annotated_frame, latest_caption, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display live video
    cv2.imshow("YOLOv8 + Caption Demo", annotated_frame)

    # Keypress events
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # Manual snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"yolo_manual_{timestamp}.jpg")
        cv2.imwrite(filename, annotated_frame)
        print(f"Manual snapshot saved: {filename}")

        # Generate caption
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image_pil, return_tensors="pt").to(device)
        out = caption_model.generate(**inputs)
        latest_caption = processor.decode(out[0], skip_special_tokens=True)
        print("Caption:", latest_caption)

cap.release()
cv2.destroyAllWindows()
