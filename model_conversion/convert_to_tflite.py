import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("digit_recognizer.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("digit_recognizer.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TensorFlow Lite!")