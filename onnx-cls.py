from ultralytics import YOLO
import torch

def export_yolo_to_onnx(model_path, output_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Get the image size from the model
    if hasattr(model.model, 'args') and hasattr(model.model.args, 'imgsz'):
        img_size = model.model.args.imgsz
    else:
        # Default to 640 if we can't determine the size
        img_size = 640
        print(f"Couldn't determine image size from model. Using default: {img_size}")

    print(f"Using image size: {img_size}")

    # Export the model to ONNX
    success = model.export(format="onnx", imgsz=img_size, opset=12)
    
    if success:
        print(f"Model successfully exported to {output_path}")
    else:
        print("Export failed")

if __name__ == "__main__":
    model_path = "yolov8n-cls.pt"  # Path to your YOLOv8 PyTorch model
    output_path = "yolov8n-cls.onnx"  # Desired output path for ONNX model

    export_yolo_to_onnx(model_path, output_path)