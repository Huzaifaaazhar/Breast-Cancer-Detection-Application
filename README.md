# Breast Cancer Detection Application

## Project Overview

Breast cancer is a critical health issue affecting millions of women worldwide. Early detection and diagnosis are crucial for improving patient outcomes and survival rates. This project aims to develop a machine learning-based application for breast cancer detection using a variety of classification algorithms to classify tumors as malignant or benign based on features extracted from breast cancer cell data.

## Business Problem

Breast cancer is one of the most common cancers among women globally, with an increasing incidence rate. Early and accurate diagnosis can significantly enhance treatment success and patient survival. Traditional methods of diagnosis involve manual inspection and analysis, which can be time-consuming and prone to human error. There is a need for a robust, automated system that can quickly and accurately classify tumors to support medical professionals in making informed decisions.

## Requirements

### Functional Requirements
1. **Data Handling**: The system should be able to load and preprocess breast cancer data from a CSV file.
2. **Feature Extraction**: Extract relevant features from the dataset to be used for model training.
3. **Model Training**: Implement various machine learning models to classify tumors as malignant or benign. Models to be used include:
   - Support Vector Classifier (SVC)
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
   - Decision Tree
   - Random Forest
   - AdaBoost
   - XGBoost
4. **Model Evaluation**: Evaluate models using accuracy metrics and confusion matrices.
5. **Model Deployment**: Deploy the final model into a Flask web application that allows users to input data and receive predictions.

### Non-Functional Requirements
1. **Scalability**: The application should handle large datasets efficiently.
2. **Usability**: The web interface should be user-friendly and accessible.
3. **Performance**: The system should provide accurate predictions quickly.

## Solution

### Data Preprocessing
The dataset used for this project contains features related to breast cancer cell characteristics and a target variable indicating whether the tumor is malignant or benign. We preprocess the data by scaling features and splitting it into training and testing sets.

### Model Implementation
We implemented several machine learning models to classify breast cancer tumors:
- **Support Vector Classifier (SVC)**: A model that finds the hyperplane that best separates classes.
- **K-Nearest Neighbors (KNN)**: A model that classifies data based on the majority vote of its neighbors.
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **Decision Tree**: A model that makes decisions based on feature splits.
- **Random Forest**: An ensemble of decision trees that improves classification performance.
- **AdaBoost**: A boosting algorithm that combines weak classifiers to form a strong classifier.
- **XGBoost**: An optimized gradient boosting algorithm for high performance and accuracy.

### Deployment
The final model is deployed using Flask, a lightweight web framework. The application allows users to upload data, preprocess it, and receive predictions from the trained model. The directory structure of the Flask application is as follows:
- `__pycache__`: Contains compiled Python files.
- `static/`: Contains static files like images.
- `templates/`: Contains HTML templates (e.g., `index.html`).
- `dataset/`: Contains the CSV dataset file.
- `app.py`: The main Flask application file.
- `model.ipynb`: Jupyter notebook with model training and evaluation.
- `requirements.txt`: Lists the Python packages required.
- `breast_cancer_detection.pickle`: The trained model saved as a pickle file.

### Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/Breast-Cancer-Detection-Application.git
   ```

2. **Navigate to the Project Directory**
   ```bash
   cd Breast-Cancer-Detection-Application
   ```

3. **Set Up the Environment**
   Ensure you have all dependencies installed. You can use a virtual environment and install required packages from `requirements.txt`:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   pip install -r requirements.txt
   ```

4. **Run the Flask Application**
   ```bash
   python app.py
   ```

5. **Access the Web Interface**
   Open a web browser and navigate to `http://127.0.0.1:5000/` to use the application.

## Contributing

Feel free to contribute to this project by submitting issues, feature requests, or pull requests.
