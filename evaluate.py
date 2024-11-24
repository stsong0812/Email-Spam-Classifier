from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
import pandas as pd
from joblib import load

def evaluate_model(model:MultinomialNB,vectorizer:TfidfVectorizer,test_set_csv='cleaned_emails_test.csv'):

    # load test samples and vectorize it
    data = pd.read_csv(test_set_csv)
    x_unvectorized = data['content'].fillna('')
    x=vectorizer.transform(x_unvectorized)
    y=data['label'].values

    # make predictions with model
    predictions=model.predict(x)

    # calculate confusion matrix, accuracy and precision based on predictions made
    matrix=confusion_matrix(y,predictions,labels=['spam','ham'])
    accuracy=accuracy_score(y,predictions)
    precision=precision_score(y,predictions,pos_label='spam')

    print(matrix)
    print(f'accuracy: {accuracy} ({accuracy*100}%)')
    print(f'precision(positive=spam): {precision} ({precision*100}%)\n')

trained_model=load('trained_model.joblib')
trained_model_vectorizer=load('trained_model_vectorizer.joblib')
print('evaluation for training.py:')
evaluate_model(trained_model,trained_model_vectorizer)

cross_trained_model=load('cross_trained_model.joblib')
cross_trained_model_vectorizer=load('cross_trained_model_vectorizer.joblib')
print('evaluation for cross_training.py:')
evaluate_model(cross_trained_model,cross_trained_model_vectorizer)