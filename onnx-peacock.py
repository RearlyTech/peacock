import cv2
import numpy as np
import onnxruntime as ort
import time
import requests

def download_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to download ImageNet labels")

def preprocess_image(img, input_size):
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def run_peacock_detection():
    # Load ONNX model
    model_path = "yolov8n-cls.onnx"
    providers = ['CPUExecutionProvider']
    if 'OpenVINOExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'OpenVINOExecutionProvider')
    session = ort.InferenceSession(model_path, providers=providers)

    # Get model details
    input_details = session.get_inputs()[0]
    output_details = session.get_outputs()[0]
    input_name = input_details.name
    output_name = output_details.name
    input_shape = input_details.shape
    output_shape = output_details.shape

    input_size = input_shape[2]  # Assuming square input
    num_classes = output_shape[1]

    print(f"Model input shape: {input_shape}")
    print(f"Model output shape: {output_shape}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Number of classes: {num_classes}")

    # Download ImageNet labels
    print("Downloading ImageNet labels...")
    labels = download_imagenet_labels()
    print(f"Number of labels loaded: {len(labels)}")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Define peacock-related classes
    peacock_classes = ['peacock']

    print("Monitoring webcam for peacocks. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            input_data = preprocess_image(frame, input_size)
            outputs = session.run([output_name], {input_name: input_data})
            scores = outputs[0][0]
            top_5_indices = np.argsort(scores)[-5:][::-1]

            for idx in top_5_indices:
                class_name = labels[idx]
                confidence = scores[idx]
                if class_name.lower() in peacock_classes and confidence > 0.45:
                    print(f"PEACOCK DETECTED: {class_name} (Confidence: {confidence:.2f})")
                else:
                    print(f"Detected: {class_name} (Confidence: {confidence:.2f})")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()

if __name__ == "__main__":
    run_peacock_detection()