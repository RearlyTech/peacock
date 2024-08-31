import torch
from ultralytics import YOLO
import cv2
import sys
import os
from contextlib import contextmanager
import time

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class SilentCallback:
    def __call__(self, info):
        pass

def run_webcam_classification():
    # Load the YOLOv8n-cls model silently
    with suppress_stdout_stderr():
        model = YOLO('yolov8n-cls.pt')
        model.add_callback("on_predict_start", SilentCallback())

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    # Get the class names
    class_names = model.names

    # Define peacock-related classes (you may need to adjust these based on the actual ImageNet classes)
    peacock_classes = ['peacock', 'peafowl']

    print("Monitoring webcam for peacocks. Press Ctrl+C to stop.")

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Run YOLOv8-cls inference on the frame silently
            with suppress_stdout_stderr():
                results = model(frame, verbose=False)[0]

            # Get the top prediction
            top_pred = results.probs.top1
            top_prob = results.probs.top1conf.item()

            # Get the class name
            class_name = class_names[top_pred]

            # Check if the prediction is a peacock and above 45% confidence
            if class_name.lower() in peacock_classes and top_prob > 0.45:
                print(f"PEACOCK DETECTED: {class_name} (Confidence: {top_prob:.2f})")

            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        # Release the webcam
        cap.release()

if __name__ == "__main__":
    run_webcam_classification()