Digit Recognition using CNN
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

Features
Built using TensorFlow and Keras

High accuracy on MNIST handwritten digits

Includes training, evaluation, and prediction scripts

Dataset
MNIST Dataset
60,000 training images + 10,000 testing images of handwritten digits (0–9).

Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/venkatesh0029/Digit-Recognition-using-CNN.git
cd Digit-Recognition-using-CNN

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Usage
1. Train the Model
bash
Copy
Edit
python train.py
This will train the CNN on the MNIST dataset and save the trained model.

2. Evaluate the Model
bash
Copy
Edit
python evaluate.py
This will load the trained model and evaluate it on the test data.

3. Make Predictions
bash
Copy
Edit
python predict.py path_to_your_image
Replace path_to_your_image with the path of a digit image you want to predict.

Model Architecture
Convolutional layers with ReLU activation

MaxPooling layers

Dropout layers to reduce overfitting

Dense (Fully Connected) layers

Output Softmax layer (10 classes: digits 0–9)

Results
Achieved about 99% accuracy on the MNIST test dataset.

License
This project is licensed under the MIT License.

