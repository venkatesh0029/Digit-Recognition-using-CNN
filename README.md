Digit Recognition using CNN
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is trained to classify grayscale images of digits (0–9) with high accuracy.​

Features
Utilizes TensorFlow and Keras for building and training the CNN model.

Achieves high accuracy on the MNIST test dataset.

Includes data preprocessing, model training, evaluation, and prediction steps.​

Dataset
The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits.​

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/venkatesh0029/Digit-Recognition-using-CNN.git
cd Digit-Recognition-using-CNN
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Train the model:

bash
Copy
Edit
python train.py
This script will load the MNIST dataset, preprocess the data, define the CNN architecture, train the model, and save the trained model to disk.

Evaluate the model:

bash
Copy
Edit
python evaluate.py
This script will load the saved model and evaluate its performance on the test dataset.

Make predictions:

bash
Copy
Edit
python predict.py path_to_image
Replace path_to_image with the path to a grayscale image of a handwritten digit. The script will output the predicted digit.

Model Architecture
The CNN model consists of the following layers:​

Convolutional layers with ReLU activation

MaxPooling layers

Dropout layers to prevent overfitting

Fully connected (Dense) layers

Output layer with softmax activation for classification​

Results
After training, the model achieves an accuracy of approximately 99% on the MNIST test dataset.​

License
This project is licensed under the MIT License. See the LICENSE file for details.
