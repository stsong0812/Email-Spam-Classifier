# Import necessary libraries
import pandas as pd  # Import pandas for data manipulation and analysis
# Import Multinomial Naive Bayes classifier from sklearn
from sklearn.naive_bayes import MultinomialNB
# Import accuracy_score to evaluate model performance
from sklearn.metrics import accuracy_score
# Import numpy for numerical operations (not used in this script but often useful)
import numpy as np

# Load the vectorized training and testing data from CSV files
# Load the training feature set (vectorized email texts)
X_train = pd.read_csv('X_train_vectorized.csv')
# Load the testing feature set (vectorized email texts)
X_test = pd.read_csv('X_test_vectorized.csv')

# Load the labels for training and testing data from CSV files
# Load training labels and flatten the array to 1D
y_train = pd.read_csv('y_train.csv').values.ravel()
# Load testing labels and flatten the array to 1D
y_test = pd.read_csv('y_test.csv').values.ravel()

# Initialize the Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()  # Create an instance of the Naive Bayes classifier

# Train the classifier using the training data
# Fit the model on the training data (X_train) and corresponding labels (y_train)
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set using the trained model
# Predict labels for the test data (X_test)
predictions = nb_classifier.predict(X_test)

# Save predictions to CSV file for evaluation
pd.DataFrame(predictions).to_csv('predictions.csv', index=False)

# Output the predictions to the console
# Print the predicted labels for the test set
print("Predictions:", predictions)

# Calculate the accuracy of the model by comparing predicted labels with actual labels
# Compute the accuracy score using the true labels (y_test) and predicted labels
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)  # Print the accuracy of the model to the console