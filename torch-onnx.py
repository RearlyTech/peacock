from ultralytics import YOLO
import torch
import torch.onnx

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

    # Create a dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Export the model
    try:
        torch.onnx.export(model.model, 
                          dummy_input, 
                          output_path, 
                          opset_version=12, 
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        print(f"Model successfully exported to {output_path}")
    except Exception as e:
        print(f"Export failed: {str(e)}")

if __name__ == "__main__":
    model_path = "yolov8n-cls.pt"  # Path to your YOLOv8 PyTorch model
    output_path = "yolov8n-cls.onnx"  # Desired output path for ONNX model

    export_yolo_to_onnx(model_path, output_path)