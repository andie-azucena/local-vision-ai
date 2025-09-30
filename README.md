# Local Vision AI Demo

This repository is POC of Computer Vision or Vision AI using a USB webcam (Logitech C270).

## Install Python & Libraries
Make sure Python 3.8+ is installed. Install the following libraries:

```
pip install opencv-python
pip install opencv-python-headless
pip install numpy
pip install ultralytics
```

Once all libraries are installed, run the Python code.

## Test Webcam

Run the `webcam-test.py` code. This should open a small window and capture wherever the webcam is pointed at. Press `q` to quit.

```
python webcam-test.py
```

If webcam window pops up but is black, change the camera index:

```
cap = cv2.VideoCapture(1)  # or 2
```

## Simple Face Detection

Run the `simple-face-detection.py` code. If the webcam sees a face, a green box will appear around the face. Press `q` to quit.

```
python simple-face-detection.py
```

## Face + Eye Detection

Run the `face-eye-detection.py` code. If the webcam sees a face, a green box will appear around the face and blue box will appear around the eye. Press `q` to quit.

```
python face-eye-detection.py
```

Use the `face-eye-detection-with-snapshot.py` if you want to take a snapshot of what the webcam captured. Snapshots are automatically captured every 5 seconds. You can also manually take a snapshot by pressing `s`. The image file is saved under the snapshots folder.

## Object Detection using YOLOv8

Run the `full-object-detection.py` code. A box with a label appears around the object. Press `q` to quit.

```
python full-object-detection.py
```

## Object Detection with Image Captioning using YOLOv8 and BLIP

Run the `object-detection-image-captioning.py` code. A box with a label appears around the object. This uses the code in full object detection using YOLOv8 then adds BLIP to give an image caption whenever you take a snapshot (press `s`). Press `q` to quit.

```
python object-detection-image-captioning.py
```

## License

This project is licensed under the MIT License.