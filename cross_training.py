# training.py

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
# Import the Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
# Import metrics for evaluation
from sklearn.metrics import accuracy_score, classification_report
# Import cross-validation scoring method
from sklearn.model_selection import cross_val_score
# Import the function to load and vectorize data from vectorize.py
from vectorize import load_and_vectorize_data

# Load and vectorize the data
X_vec, y, vectorizer = load_and_vectorize_data()
# X_vec: The vectorized features (email content transformed into numerical format)
# y: The labels (0 for non-spam, 1 for spam)
# vectorizer: The TfidfVectorizer instance used for transforming the text data

# Initialize the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()  # Create an instance of the Naive Bayes classifier

# Perform cross-validation
cv_scores = cross_val_score(nb_classifier, X_vec, y, cv=5)
# cv=5 indicates that we are using 5-fold cross-validation
# This means the dataset will be split into 5 parts, and the model will be trained and evaluated 5 times,
# each time using a different part as the test set and the remaining parts as the training set.

# Print the cross-validation scores for each fold
print("Cross-validation scores:", cv_scores)
# Print the mean cross-validation score, which provides an overall performance metric across all folds
print("Mean cross-validation score:", cv_scores.mean())

# Fit the model on the entire dataset (optional)
nb_classifier.fit(X_vec, y)
# This step trains the model on the entire dataset.
# It is optional because we have already evaluated the model using cross-validation.

# Make predictions on the entire dataset (optional)
predictions = nb_classifier.predict(X_vec)
# Use the trained model to predict labels for the vectorized features

# Output the classification report
print(classification_report(y, predictions))
# This report includes various metrics such as precision, recall, F1-score, and support,
# providing a detailed evaluation of the model's performance on the dataset.
