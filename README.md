# Credit Card Fraud Detection using Logistic Regression
This project develops a machine learning model to detect fraudulent credit card transactions using logistic regression. Fraud detection is critical in reducing financial losses and improving security for users and financial institutions.

# Project Overview
Credit card fraud detection involves identifying potentially fraudulent transactions from massive volumes of transaction data. This project applies logistic regression, a supervised machine learning algorithm, to classify transactions as either legitimate or fraudulent. The main challenges addressed include handling imbalanced data, ensuring model accuracy, and optimizing precision and recall metrics to minimize false positives and false negatives.

# Objectives
Detect fraudulent transactions effectively and with high precision.
Handle imbalanced data to improve model performance on the minority class (fraud cases).
Create a scalable and interpretable model that can be used as a baseline for more complex algorithms.
# Dataset
The dataset used is the Credit Card Fraud Detection Dataset from Kaggle, which contains anonymized credit card transactions labeled as fraudulent or non-fraudulent. This dataset is highly imbalanced, with only a small fraction of transactions labeled as fraud. Here is the dataset link
      "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"

Features: 30 features, including Time, Amount, and anonymized variables V1 to V28.
Target: Class (1 indicates a fraudulent transaction, 0 indicates a legitimate transaction).
# Project Workflow
Data Exploration and Preprocessing: Analyzing feature distributions, handling missing values, and normalizing Amount and Time.
Data Balancing Techniques: Applying techniques like SMOTE (Synthetic Minority Oversampling Technique) to manage class imbalance.
Model Training: Implementing logistic regression to classify transactions.
Evaluation: Using performance metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.
# Tools and Libraries
Python: Programming language used for data processing and modeling.
Pandas, Numpy: Data manipulation libraries.
Scikit-learn: Machine learning library used for model implementation and evaluation.
Matplotlib, Seaborn: Visualization libraries for Exploratory Data Analysis (EDA).
# Key Results
The logistic regression model achieved a strong balance between precision and recall, making it effective at detecting fraudulent transactions with minimal false positives and false negatives.

# Future Enhancements
Model Optimization: Tune hyperparameters to improve model performance.
Advanced Models: Explore more complex models like Random Forest, Gradient Boosting, or Neural Networks.
Deployment: Implement real-time fraud detection for scalable deployment.
