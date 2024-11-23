from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
from scipy.sparse import csr_matrix
import pandas as pd

def evaluate_model(model:MultinomialNB):

    # load test samples and it's labels
    test_samples=csr_matrix(pd.read_csv('X_test_vectorized.csv'))
    actual_labels=pd.read_csv('y_test.csv')

    # make predictions with model
    predictions=model.predict(test_samples)

    # calculate confusion matrix, accuracy and precision based on predictions made
    matrix=confusion_matrix(actual_labels,predictions,labels=['spam','ham'])
    accuracy=accuracy_score(actual_labels,predictions)
    precision=precision_score(actual_labels,predictions,pos_label='spam')

    print(matrix)
    print(f'accuracy: {accuracy} ({accuracy*100}%)')
    print(f'precision: {precision} ({precision*100}%)')