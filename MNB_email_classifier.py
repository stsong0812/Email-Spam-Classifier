import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump, load


# Check and download NLTK resources
def download_nltk_resources():
    print("\nChecking and downloading NLTK resources if necessary...")
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')
    print("NLTK resources ready.\n")


download_nltk_resources()


# Preprocessor function
def preprocessor(data):
    data = data.lower()
    data = re.sub(r'[^a-z\s]', '', data)
    tokens = word_tokenize(data)
    stop_words = set(stopwords.words('english')).union({'subject', 'enron'})
    tokens = [word for word in tokens if word not in stop_words and len(word) > 3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


# Load and preprocess data
def load_data(label, directory):
    print(f"\nLoading and preprocessing data from {directory}...")
    data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            email_contents = file.read()
            processed_data = preprocessor(email_contents)
            data.append({'label': label, 'content': processed_data})
    print(f"Finished processing data from {directory}.\n")
    return data


# Save preprocessed data
def preprocess_and_save(spam_path, ham_path, output_csv):
    print(f"\nPreprocessing spam and ham datasets and saving to {output_csv}...")
    spam_data = load_data('spam', spam_path)
    ham_data = load_data('ham', ham_path)
    all_data = spam_data + ham_data
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}.\n")


# Vectorize data
def vectorize_and_split(input_csv):
    print(f"\nLoading and vectorizing data from {input_csv}...")
    data = pd.read_csv(input_csv)
    data['content'] = data['content'].fillna('')
    vectorizer = CountVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(data['content'])
    X_train, X_test, y_train, y_test = train_test_split(X_vec, data['label'], test_size=0.2, random_state=42)
    dump(vectorizer, 'trained_model_vectorizer.joblib')
    print("Vectorizing complete. Saving split data...")
    pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out()).to_csv('X_train_vectorized.csv', index=False)
    pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out()).to_csv('X_test_vectorized.csv', index=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
    print("Split data saved: X_train, X_test, y_train, y_test.\n")
    return X_train, X_test, y_train, y_test, vectorizer


# Train model and save
def train_model(X, y, model_name):
    print(f"\nTraining model and saving as {model_name}...")
    model = MultinomialNB()
    model.fit(X, y)
    dump(model, model_name)
    print(f"Model saved as {model_name}.\n")
    return model


# Evaluate model
def evaluate_model(model, vectorizer, test_set_csv):
    print(f"\nEvaluating model using test set from {test_set_csv}...")
    data = pd.read_csv(test_set_csv)
    x_unvectorized = data['content'].fillna('')
    x = vectorizer.transform(x_unvectorized)
    y = data['label'].values
    predictions = model.predict(x)
    matrix = confusion_matrix(y, predictions, labels=['spam', 'ham'])
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, pos_label='spam')
    print("Confusion Matrix:")
    print(matrix)
    print(f"Accuracy: {accuracy} ({accuracy*100}%)")
    print(f"Precision (positive=spam): {precision} ({precision*100}%)\n")


# Cross-validation
def cross_validate(X, y):
    print("\nPerforming cross-validation...")
    model = MultinomialNB()
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", cv_scores)
    print(f"Mean cross-validation score: {cv_scores.mean()}\n")
    return model.fit(X, y)


# Main function
def main():
    print("\nStarting email spam filter pipeline...")
    spam_path = "dataset/spam"
    ham_path = "dataset/ham"
    preprocess_and_save(spam_path, ham_path, 'cleaned_emails.csv')
    preprocess_and_save(spam_path, ham_path, 'cleaned_emails_test.csv')

    # Vectorize and split data
    X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split('cleaned_emails.csv')

    # Train and evaluate base model
    base_model = train_model(X_train, y_train, 'trained_model.joblib')
    evaluate_model(base_model, vectorizer, 'cleaned_emails_test.csv')

    # Cross-validation
    print("\nRunning cross-validation...")
    cross_model = cross_validate(X_train, y_train)
    dump(cross_model, 'cross_trained_model.joblib')
    print("Cross-validation model saved as cross_trained_model.joblib.\n")

    # Evaluate cross-trained model
    print("Evaluating cross-trained model...")
    evaluate_model(cross_model, vectorizer, 'cleaned_emails_test.csv')

    print("Email spam filter pipeline complete!\n")


if __name__ == "__main__":
    main()