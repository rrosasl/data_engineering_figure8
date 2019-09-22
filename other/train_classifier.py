# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys



import re
nltk.download(['stopwords','punkt','wordnet','averaged_perceptron_tagger'])


def load_data(db_filepath='InsertDatabaseName.db'):
    '''
    Reads database sqlite data base as a data Frame
    Separates into independent variable X and dependent variable Y

    Args:
        db_filepath(str): filepath to the database

    Returns:
        X (pandas DataFrame): Independent variable
        Y (pandas DataFrame): Dependent variable
        cat_names (list): Y category names
    '''

    engine = create_engine('sqlite:///' + db_filepath)
    df = pd.read_sql_table('messages_labeled',engine)
    df = df[df.related != 2]

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    cat_names = Y.columns.tolist()

    return X, Y

def tokenize(text):
    '''
    INPUT: Text (string)
    PROCESS:
    > Lowercase
    > word_tokenize
    > remove stopwords

    OUTPUT: Normalized list of words
    '''
    text = re.sub("[^a-zA-Z0-9]"," ",text) # remove special characters
    text = text.lower() #lowercase entire text
    words = word_tokenize(text) #Split into words

    stop_words = stopwords.words('english') # load stop words
    words = [word for word in words if word not in stop_words] #only those words not in stop_words

    #Lemmatization & stemmization

    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


def build_model():
    '''
    Returns GrindeSearchCV object as model, i.e. classifier with optimized parameters

    args:
        None

    Returns:
        cv : GridSearch Model Object

    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    #'clf__estimator__n_estimators' : [10,100],
    #'clf__estimator__max_depth' : [10,100,None],
    #'clf__estimator__min_samples_split' : [2,10],
    'clf__estimator__min_samples_leaf' : [1,2]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv

def save_model(model, model_filepath):
    """saves the model to the given filepath
    Args:
        model (scikit-learn): fitted model
        model_filepath (string): filepath
    Returns:
        None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        model_filepath = sys.argv[:-1]
        db_filepath = 'InsertDatabaseName.db'
        print('Loading data...\n    database location: {}'.format(db_filepath))
        X, Y = load_data(db_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...   model: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('You just saved the trained model')

    else:
        print('Error, file locations not correct')
