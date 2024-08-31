    import cv2
import torch
from ultralytics import YOLO

def run_webcam_classification():
    # Load the YOLOv8n-cls model
    model = YOLO('yolov8n-cls.pt')  # This will download the model if not already present

    # Open the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    # Get the class names
    class_names = model.names

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8-cls inference on the frame
        results = model(frame)[0]  # Only one image, so we take the first result

        # Get the top prediction
        top_pred = results.probs.top1
        top_prob = results.probs.top1conf.item()

        # Get the class name
        class_name = class_names[top_pred]

        # Create the text to display
        text = f"{class_name}: {top_prob:.2f}"

        # Put the text on the frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv8-cls Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_classification()