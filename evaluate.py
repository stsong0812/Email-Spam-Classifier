import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix,classification_report

# load test samples
data = pd.read_csv('cleaned_emails_test.csv')
x_unvectorized = data['content'].fillna('')
y=data['label'].values

def evaluate_model(model,vectorizer,x_unvectorized,y):

    # vectorize test samples
    x=vectorizer.transform(x_unvectorized)

    # make predictions with model
    predictions=model.predict(x)

    # calculate confusion matrix, accuracy, precision, recall, etc
    matrix=confusion_matrix(y,predictions,labels=['spam','ham'])

    print(matrix)
    print(classification_report(y,predictions,digits=4))

trained_model=load('trained_model.joblib')
trained_model_vectorizer=load('trained_model_vectorizer.joblib')
print('evaluation for training.py:')
evaluate_model(trained_model,trained_model_vectorizer,x_unvectorized,y)

cross_trained_model=load('cross_trained_model.joblib')
cross_trained_model_vectorizer=load('cross_trained_model_vectorizer.joblib')
print('evaluation for cross_training.py:')
evaluate_model(cross_trained_model,cross_trained_model_vectorizer,x_unvectorized,y)