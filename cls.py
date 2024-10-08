import cv2
import torch
from ultralytics import YOLO
import sys
import os
from contextlib import contextmanager
import time
import threading
import subprocess

# GPIO control functions
def set_gpio_mode(pin, mode):
    subprocess.run(["gpio", "mode", str(pin), mode])

def write_gpio(pin, value):
    subprocess.run(["gpio", "write", str(pin), str(value)])

# Function to blink the LED for 5 seconds
def blink_led(pin, duration=5):
    end_time = time.time() + duration
    while time.time() < end_time:
        write_gpio(pin, 1)  # Turn LED on
        time.sleep(0.1)     # 100 ms delay
        write_gpio(pin, 0)  # Turn LED off
        time.sleep(0.1)     # 100 ms delay

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

    # Set GPIO pin 2 mode to output
    set_gpio_mode(2, "out")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8-cls inference on the frame silently
        with suppress_stdout_stderr():
            results = model(frame, verbose=False)[0]

        # Get the top prediction
        top_pred = results.probs.top1
        top_prob = results.probs.top1conf.item()

        # Get the class name
        class_name = class_names[top_pred]

        # Create a clean copy of the frame
        display_frame = frame.copy()

        # Check if the prediction is a peacock and above 45% confidence
        if class_name.lower() in peacock_classes and top_prob > 0.45:
            text = f"PEACOCK DETECTED: {class_name} ({top_prob:.2f})"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(text)  # This is the only print statement that should appear

            # Start the LED blinking in a separate thread for 5 seconds
            threading.Thread(target=blink_led, args=(2, 5)).start()

        # Display the frame
        cv2.imshow("YOLOv8-cls Inference", display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_classification()
