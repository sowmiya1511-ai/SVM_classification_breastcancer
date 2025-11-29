**Features

1 Exploratory Data Analysis (EDA)

2 SVM Model Training & Evaluation

3 Accuracy, Confusion Matrix, and Classification Report

4 Interactive Streamlit Application for prediction

5 Model saved using pickle for deployment

**Project Structure
├── data/
│   └── breast_cancer.csv        
├── models/
│   └── svm_model.pkl           
├── app/
│   └── streamlit_app.py        
├── notebooks/
│   └── model_training.ipynb     
├── requirements.txt
└── README.md

**Install dependencies
pip install -r requirements.txt

** How the Model Works

Load dataset

Preprocess features (scaling)

Train SVM with RBF kernel

Evaluate model

Save trained model to svm_model.pkl

Streamlit app loads the model and takes user input

App predicts benign or malignant

**Tech Stack

Python

scikit-learn

NumPy, Pandas

Matplotlib/Seaborn

Streamlit

Pickle

<img width="1223" height="775" alt="Image" src="https://github.com/user-attachments/assets/c3192fc6-9602-434f-afc6-df02da1a2af0" />
<img width="1110" height="412" alt="Image" src="https://github.com/user-attachments/assets/2051cce9-f634-40cb-8e0e-356d57dda525" />
