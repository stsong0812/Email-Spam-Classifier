# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
# For text vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split
from joblib import dump


def load_and_vectorize_data():
    """
    Load and vectorize email data from a CSV file.

    This function reads preprocessed email data from a CSV file, handles any NaN values
    in the content column, and uses the TF-IDF vectorization technique to convert the
    email text into a numerical format suitable for machine learning models.

    Returns:
        X_vec (scipy.sparse.csr.csr_matrix): A sparse matrix of shape (n_samples, n_features)
            containing the TF-IDF representation of the email content.
        labels (numpy.ndarray): An array of shape (n_samples,) containing the labels
            (0 for non-spam and 1 for spam) corresponding to the email content.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer object that can be used
            for transforming new data.
    """

    # Load preprocessed data from a CSV file
    data = pd.read_csv('cleaned_emails.csv')

    # Handle NaN values in the 'content' column by filling them with an empty string
    # This ensures that the vectorizer does not encounter any NaN values during transformation
    data['content'] = data['content'].fillna('')

    # Initialize the TF-IDF vectorizer
    # max_features=5000 limits the number of features to the 5000 most important words
    # ngram_range=(1, 1) specifies that only unigrams (single words) will be used
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))

    # Fit the vectorizer to the content data and transform it into a TF-IDF representation
    # This step learns the vocabulary and idf from the training data and converts the text
    # into a numerical format that machine learning algorithms can process
    X_vec = vectorizer.fit_transform(data['content'])

    # Return the vectorized data (X_vec), the labels, and the fitted vectorizer
    return X_vec, data['label'].values, vectorizer


# Load preprocessed data
# Assuming `cleaned_emails.csv` contains two columns:
# 'label' - where 0 indicates non-spam and 1 indicates spam
# 'content' - which contains the actual email text
# After loading the data and before splitting it into train and test sets
data = pd.read_csv('cleaned_emails.csv')

# Check for NaN values in the 'content' column
# print("NaN values in the 'content' column:", data['content'].isna().sum())

# Option 1: Drop rows with NaN values
# data = data.dropna(subset=['content'])

# Option 2: Fill NaN values with an empty string (uncomment if you prefer this method)
data['content'] = data['content'].fillna('')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['content'],  # Features (email text)
    data['label'],    # Labels (spam or non-spam)
    test_size=0.2,    # 20% of data for testing
    random_state=42   # Seed for reproducibility
)
# Choose a vectorizer to convert text data into numerical format
# Here, we are using TfidfVectorizer which transforms text to feature vectors
# max_features=5000 limits the number of features to 5000 most important words
# stop_words='english' removes common English stop words (e.g., 'the', 'is')
# ngram_range=(1, 1) means we are using unigrams (single words); can be changed to (1, 2) for unigrams and bigrams
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to 5000 features
    # stop_words='english',  # Remove common English stop words
    ngram_range=(1, 1)    # Use unigrams only (single words)
)

# Fit the vectorizer to the training data and transform it
# This learns the vocabulary and idf from the training data and transforms it into a TF-IDF representation
X_train_vec = vectorizer.fit_transform(X_train)

# Transform the testing data using the same vectorizer
# This uses the vocabulary learned from the training data to transform the test data
X_test_vec = vectorizer.transform(X_test)

# Convert the sparse matrices to DataFrames for easier saving and manipulation
# The toarray() method converts the sparse matrix to a dense format
# get_feature_names_out() retrieves the feature names (words) from the vectorizer
X_train_df = pd.DataFrame(X_train_vec.toarray(
), columns=vectorizer.get_feature_names_out())
X_test_df = pd.DataFrame(X_test_vec.toarray(
), columns=vectorizer.get_feature_names_out())

# Save the vectorized training and testing data to CSV files
# This allows for easy loading later for model training and evaluation
# Save training features
X_train_df.to_csv('X_train_vectorized.csv', index=False)
# Save testing features
X_test_df.to_csv('X_test_vectorized.csv', index=False)

# Save the labels for training and testing data separately
# This is important for model training and evaluation
y_train.to_csv('y_train.csv', index=False)  # Save training labels
y_test.to_csv('y_test.csv', index=False)    # Save testing labels

# save training.py model's vectorizer
dump(vectorizer,'trained_model_vectorizer.joblib')