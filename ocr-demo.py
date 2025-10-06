import cv2
import easyocr
from datetime import datetime
import os
import time

# Initialize EasyOCR reader (English only)
reader = easyocr.Reader(['en'], gpu=False)

# Create folder for snapshots
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save current frame + OCR results.")
print("Press 'q' to quit.")

# Set a processing interval (in seconds) to avoid lag
last_ocr_time = 0
ocr_interval = 1  # seconds between OCR updates
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed (optional)
    display_frame = cv2.resize(frame, (640, 480))

    # Run OCR every 1 second (to keep performance smooth)
    current_time = time.time()
    if current_time - last_ocr_time > ocr_interval:
        last_results = reader.readtext(display_frame, paragraph=False, detail=1)
        last_ocr_time = current_time

    # Draw bounding boxes for each detected text
    extracted_text = ""
    for (bbox, text, conf) in last_results:
        if conf > 0.5:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(display_frame, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            extracted_text += text + " "

    # Display the annotated feed
    cv2.imshow("Live OCR Demo (EasyOCR)", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = os.path.join(output_dir, f"ocr_live_{timestamp}.jpg")
        txt_file = os.path.join(output_dir, f"ocr_live_{timestamp}.txt")

        cv2.imwrite(img_file, display_frame)
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(extracted_text.strip())

        print(f"Saved snapshot: {img_file}")
        print(f"OCR text: {txt_file}")
        print("Detected text:", extracted_text.strip())

cap.release()
cv2.destroyAllWindows()
