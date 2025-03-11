# TinyML Digit Recognizer

## Overview
This project is a TinyML-based **handwritten digit recognizer** using TensorFlow Lite. It is trained on the MNIST dataset and optimized for lightweight deployment. The model can be used on edge devices or run locally using a Python script.

## Features
✅ Trained using **Google Colab** on the **MNIST dataset**  
✅ Saved in **.h5 format**, then converted to **TensorFlow Lite (.tflite)**  
✅ Runs using **Python script** without extra hardware  
✅ Uses **Laptop Webcam** for real-time digit recognition  

## Project Structure
```
tinyml-digit-recognizer/
│── model_training/
│   ├── train_model.ipynb        # Google Colab Notebook for training
│   ├── digit_recognizer.h5      # Saved Keras model
│── model_conversion/
│   ├── convert_to_tflite.py     # Script to convert .h5 to .tflite
│   ├── digit_recognizer.tflite  # Converted TensorFlow Lite model
│── model_inference/
│   ├── handwritten_recognition.py  # Script to run the TinyML model
│── README.md                    # Project documentation
|── LICENSE                      # MIT License
│── requirements.txt             # Required Python libraries
│── .gitignore                    # Ignore unnecessary files
```

## Setup Instructions
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prateek-dhanker/tinyml-digit-recognizer.git
cd tinyml-digit-recognizer
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional)
Run the Jupyter Notebook in `model_training/` to train the model.

### 4️⃣ Convert to TensorFlow Lite (If Needed)
```bash
python model_conversion/convert_to_tflite.py
```

### 5️⃣ Run the Model for Digit Recognition
```bash
python model_inference/run_model.py
```

## Dependencies
- Python 3.x
- TensorFlow
- NumPy
- OpenCV (if using webcam input)
- Matplotlib (optional for visualization)

## Example Output
```
Predicted Digit: 7
Confidence Score: 98.5%
```

## Contributing
Feel free to fork this repository and contribute! Suggestions and improvements are welcome.

## License
This project is open-source and available under the MIT License.

---