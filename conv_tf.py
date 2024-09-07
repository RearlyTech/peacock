import onnx
import tensorflow as tf
import tf2onnx

def convert_onnx_to_tflite(onnx_model_path, tflite_model_path):
    # Load ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    # Convert ONNX model to TensorFlow
    tf_rep = tf2onnx.backend.prepare(onnx_model)
    
    # Get input and output details
    input_shape = tf_rep.inputs[0].shape
    output_shape = tf_rep.outputs[0].shape
    
    # Create a Keras model
    inputs = tf.keras.Input(shape=input_shape[1:])
    outputs = tf.keras.layers.Lambda(lambda x: tf_rep.run(x))(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save Keras model
    tf_model_path = "tf_model"
    keras_model.save(tf_model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {tflite_model_path}")

# Usage
onnx_model_path = 'yolov8n-cls.onnx'
tflite_model_path = 'yolov8n-cls.tflite'
convert_onnx_to_tflite(onnx_model_path, tflite_model_path)