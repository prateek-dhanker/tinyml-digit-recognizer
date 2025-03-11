import cv2
import numpy as np
import tensorflow as tf
import os


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_conversion/digit_recognizer.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))  # Resize to match MNIST format
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1).astype(np.float32)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)  # Draw box for input
    roi = frame[100:300, 100:300]  # Extract region of interest
    
    # Preprocess and run inference
    input_data = preprocess_image(roi)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    digit = np.argmax(output_data)
    
    # Display result
    cv2.putText(frame, f"Prediction: {digit}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Digit Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()