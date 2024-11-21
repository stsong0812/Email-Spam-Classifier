import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# IMPORTANT!

# Used dataset (ensure they are in the working directory)
# https://drive.google.com/file/d/13g8D4KxHoS0iZPCHSHo93sFmHKKcmRId/view?usp=sharing

# Downloads nltk resources needed, can be commented after download
# nltk.download('stopwords')  # Downloads list of common stopwords
# nltk.download('punkt')      # Downloads resources needed for tokenization


def preprocessor(data):
    '''
    Preprocessor function, takes in data from the load_data function.
    Raw data is pre-processed for use in vectorization function.
    '''
    data = data.lower()     # Lowercases raw data
    # Removes non-alphanumeric characters (preserves whitespace)
    data = re.sub(r'[^a-z\s]', '', data)
    # Tokenize data using nltk's word_tokenize function
    tokens = word_tokenize(data)
    # Initialize stopwords from downloaded nltk stopwords list
    stop_words = set(stopwords.words('english'))
    # Include extra stopwords not included in nltk's list
    stop_words.update(['subject', 'enron'])
    # Filter out declared stopwords and words with 3 or fewer characters
    tokens = [
        word for word in tokens if word not in stop_words and len(word) > 3]
    stemmer = PorterStemmer()   # Initialize instance of nltk's PorterStemmer class
    # Apply stemming to each token
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)     # Return tokens in a single string


def load_data(label, directory):
    '''
    Function to load raw data from ham or spam directories, must be provided
    a label (ham or spam) and the directory path.

    Iterates through files in the provided path, reads the raw data, and calls
    the preprocessor function and appends the returned preprocessed data to a list.
    '''
    data = []   # Initialize empty data list to append to
    # Loop through files in directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Opens iterated through files
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            # Read file content to email_contents to call preprocessor function
            email_contents = file.read()
            # Call preprocessor function with raw email contents
            processed_data = preprocessor(email_contents)
            # Append preprocessed data to our data list
            data.append({'label': label, 'content': processed_data})
    return data


# Declare both spam and ham os paths
spam_path = "dataset/spam"
ham_path = "dataset/ham"

# Call load_data function for spam and ham
spam_data = load_data('spam', spam_path)
ham_data = load_data('ham', ham_path)

# Combine spam and ham data
all_data = spam_data + ham_data

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
df.to_csv('cleaned_emails.csv', index=False)
