# Credit_Default_Prediction_using_RandomForestModel
Credit Card Default Prediction using Machine Learning Models
This repository contains a project focused on building a machine learning model for credit card default prediction. The objective is to develop a reliable predictive model that can accurately determine whether a credit card holder is likely to default on their payments. Three different machine learning models, namely K-nearest neighbors (KNN), logistic regression, and random forest, have been implemented and compared in terms of their accuracy scores.

Dataset
The dataset used in this project has been sourced from the ybifoundation/Dataset repository. It provides a comprehensive set of features related to credit card holders, such as demographic information, transaction history, and credit history. This dataset has been preprocessed and carefully prepared to ensure its quality and reliability.

Data Preprocessing
Before building the predictive models, the dataset has been divided into training and testing subsets. This division allows us to evaluate the models' performance on unseen data and assess their generalization capabilities.

Model Selection
Three different machine learning models have been selected for credit card default prediction:

1. Logistic Regression
Logistic regression is a widely used statistical model for binary classification problems. In this project, logistic regression was implemented to build a predictive model for credit card default. The logistic regression model achieved an accuracy score of 90% on the testing dataset. Logistic regression utilizes a best-fitted line to separate the two classes (default or non-default) based on the given features.

2. K-nearest Neighbors (KNN)
K-nearest neighbors (KNN) is a non-parametric algorithm that makes predictions based on the closest k neighbors in the feature space. In our project, the KNN model achieved an accuracy score of 83% on the testing dataset. KNN considers the similarity of the feature vectors to determine the class label of a given sample.

3. Random Forest
Random forest is an ensemble learning method that combines multiple decision trees to make predictions. It is known for its ability to handle complex relationships between variables and provide robust predictions. In this project, the random forest model achieved the best result, with an accuracy score of 100% on the testing dataset.

Instructions
To use this repository, follow these steps:

Clone or download the repository to your local machine.
Ensure that the necessary dependencies are installed (e.g., Python, scikit-learn, etc.).
Open the provided Jupyter Notebook or Python script that contains the implementation of the machine learning models.
Execute the code to train and evaluate the models on the provided training and testing datasets.
Compare the accuracy scores of the logistic regression, KNN, and random forest models to assess their performance in credit card default prediction.
Feel free to explore further enhancements or modifications to improve the accuracy and performance of the models. Additionally, consider conducting further analysis and evaluation metrics to gain deeper insights into the predictive capabilities of the models.
