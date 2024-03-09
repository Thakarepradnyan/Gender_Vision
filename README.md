# Real Time Gender Detection Using Random Forest

This project aims to demonstrate real-time gender detection using Random Forest, a machine learning algorithm. It consists of three main components: data preprocessing and model training, real-time gender detection, and evaluation. The project utilizes Python, OpenCV, scikit-learn, and other libraries for image processing and machine learning tasks.

## Project Structure

- `algorithm_path.py`: Contains the code for data preprocessing, model training, and evaluation.
- `real_time_detection.py`: Implements real-time gender detection using the trained model.
- `algorithm.ipynb`: Jupyter Notebook file containing the code used for model training and evaluation.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Real-Time-Gender-Detection
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Train the Model**:
   - Open `algorithm_path.py` and execute the code to train the gender detection model.
   - Ensure the image data is available in the specified directories
4. **Real Time Detection**:
   - Run `real_time_detection.py` to perform real-time gender detection using the trained model.
   - Make sure the `algoritma` module (model and functions) is accessible from `real_time_detection.py`.
   - Adjust paths and dependencies if necessary.
5. **Observation**:
   - A window will open displaying the real-time video feed from the camera.
   - The script will detect faces and attempt to classify gender in real-time.
   - Press 'q' to exit the real-time detection.
