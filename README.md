# Fraud Transaction Detection

This project aims to detect fraudulent transactions using machine learning models. It involves data preprocessing, model training, evaluation, and making predictions based on the best performing model. The dataset used for this project is a credit card transaction dataset that includes various transaction attributes.

## Introduction

The goal of this project is to build a machine learning model that can detect fraudulent transactions based on transaction data. The dataset includes features such as transaction amount, time, and others, along with a target variable indicating whether the transaction is fraudulent.

## Description

The notebook trains several machine learning models to detect fraudulent transactions using a dataset containing various transaction parameters. The models evaluated include:

Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Random Forest
Gradient Boosting

## Dataset

The dataset used for training and testing is preprocessed and engineered to improve the accuracy of the models. The dataset consists of several transaction predictor variables and one target variable (Class) which indicates whether a transaction is fraudulent (1) or not (0).

You can downnload the Dataset here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Installation

To run this project, you need to have Python installed along with the following libraries:

numpy
pandas
scikit-learn
matplotlib
seaborn
Jupyter Notebook

You can install these dependencies using pip: pip install numpy pandas scikit-learn matplotlib seaborn notebook

## Results

The project demonstrates the following:

Exploratory Data Analysis (EDA) to understand the dataset.
Data preprocessing steps such as handling missing values and feature engineering.
Training multiple machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, etc.).
Evaluating the performance of these models using metrics like accuracy, precision, recall, and F1-score.
Hyperparameter tuning to improve model performance.

# Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.
