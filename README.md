# Py-Cancer-Cell-Classification


This project demonstrates how to classify breast cancer cells as either 'malignant' or 'benign' using machine learning. We use the Breast Cancer Wisconsin (Diagnostic) dataset, which is included in the Scikit-learn library.

# Project Overview
The goal of this project is to build a machine learning model that can predict whether a tumor is malignant or benign based on certain features of the tumor. We use the Naive Bayes algorithm for classification, which is well-suited for binary classification problems.

# Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. It includes data on 569 instances of tumors with 30 features each, such as radius, texture, perimeter, area, and more. The dataset is preloaded in Scikit-learn and can be accessed with the following command:

```python 

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```
# Steps to Implement
1. Importing Necessary Modules
We start by importing Scikit-learn and loading the dataset:

```python 

import sklearn
from sklearn.datasets import load_breast_cancer
### 2. Loading and Organizing Data
### We load the dataset and organize it into labels and features:


data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
### 3. Splitting the Data
### We split the data into a training set and a test set using the train_test_split function:


from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)
### 4. Building and Training the Model
### We use the Naive Bayes algorithm to build and train our model:

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
### 5. Making Predictions
### We use the trained model to predict the classifications of the test data:


predictions = gnb.predict(test)
### 6. Evaluating the Model
### We evaluate the accuracy of the model by comparing the predictions with the actual labels:

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, predictions))
### 7. Output
### The accuracy of the model is approximately 94.15%, meaning it correctly classifies the tumors 94.15% of the time.
```
# Installation
To run this project, you need to install the necessary Python modules:

pip install scikit-learn
pip install jupyter

It's recommended to use Jupyter Notebook for this project to run and see the code step by step.

# Conclusion
This project shows how machine learning can be applied to real-world problems like cancer classification. By using the Scikit-learn library, we were able to build an accurate model with minimal code.

# License
This project is open-source and available under the MIT License.
