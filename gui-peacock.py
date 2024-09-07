import cv2
import torch
from ultralytics import YOLO
import sys
import os
from contextlib import contextmanager
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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

class PeacockDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = None
        self.model = None
        self.is_running = False

        # Create GUI elements
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_start = ttk.Button(window, text="Start", command=self.start)
        self.btn_start.pack(padx=10, pady=10)

        self.btn_stop = ttk.Button(window, text="Stop", command=self.stop)
        self.btn_stop.pack(padx=10, pady=10)

        self.status_label = ttk.Label(window, text="Status: Stopped")
        self.status_label.pack(padx=10, pady=10)

        # Set GPIO pin 2 mode to output
        set_gpio_mode(2, "out")

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
            self.status_label.config(text="Status: Running")

            # Load the YOLOv8n-cls model silently
            with suppress_stdout_stderr():
                self.model = YOLO('yolov8n-cls.pt')
                self.model.add_callback("on_predict_start", SilentCallback())

            self.vid = cv2.VideoCapture(self.video_source)
            self.update()

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self.status_label.config(text="Status: Stopped")

            if self.vid is not None:
                self.vid.release()
                self.vid = None

    def update(self):
        if self.is_running and self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                # Run YOLOv8-cls inference on the frame silently
                with suppress_stdout_stderr():
                    results = self.model(frame, verbose=False)[0]

                # Get the top prediction
                top_pred = results.probs.top1
                top_prob = results.probs.top1conf.item()

                # Get the class name
                class_name = self.model.names[top_pred]

                # Create a clean copy of the frame
                display_frame = frame.copy()

                # Check if the prediction is a peacock and above 45% confidence
                peacock_classes = ['peacock', 'peafowl']
                if class_name.lower() in peacock_classes and top_prob > 0.45:
                    text = f"PEACOCK DETECTED: {class_name} ({top_prob:.2f})"
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(text)  # This is the only print statement that should appear

                    # Start the LED blinking in a separate thread for 5 seconds
                    threading.Thread(target=blink_led, args=(2, 5)).start()

                # Convert the frame to RGB and then to ImageTk format
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.is_running:
            self.window.after(10, self.update)

    def on_closing(self):
        self.stop()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PeacockDetectorApp(root, "Peacock Detector")
    root.mainloop()