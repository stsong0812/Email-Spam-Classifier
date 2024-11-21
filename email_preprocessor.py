import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# IMPORTANT!

# Used dataset (ensure they are in the working directory)
# https://drive.google.com/file/d/13g8D4KxHoS0iZPCHSHo93sFmHKKcmRId/view?usp=sharing

# Downloads nltk resources needed, can be commented after download
#nltk.download('stopwords')  # Downloads list of common stopwords
#nltk.download('punkt')      # Downloads resources needed for tokenization

def preprocessor(data):
    '''
    Preprocessor function, takes in data from the load_data function.
    Raw data is pre-processed for use in vectorization function.
    '''
    data = data.lower()     # Lowercases raw data
    data = re.sub(r'[^a-z\s]', '', data)    # Removes non-alphanumeric characters (preserves whitespace)
    tokens = word_tokenize(data)    # Tokenize data using nltk's word_tokenize function
    stop_words = set(stopwords.words('english'))     # Initialize stopwords from downloaded nltk stopwords list
    stop_words.update(['subject', 'enron'])     # Include extra stopwords not included in nltk's list
    tokens = [word for word in tokens if word not in stop_words and len(word) > 3]  # Filter out declared stopwords and words with 3 or fewer characters
    stemmer = PorterStemmer()   # Initialize instance of nltk's PorterStemmer class
    tokens = [stemmer.stem(word) for word in tokens]    # Apply stemming to each token
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
            email_contents = file.read()    # Read file content to email_contents to call preprocessor function
            processed_data = preprocessor(email_contents)   # Call preprocessor function with raw email contents
            data.append({'label':label, 'content':processed_data})  # Append preprocessed data to our data list
    return data
    
# Declare both spam and ham os paths
spam_path = "dataset/spam"
ham_path = "dataset/ham"

# Call load_data function for spam and ham
load_data('spam', spam_path)
load_data('ham', ham_path)