# import statements
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet') # download for lemmatization

# Usage while in this folder:
# python train_classifier.py ../data/DisasterResponse.db classifier.pkl

def main():
    # Read in database file
    #engine = create_engine('sqlite:///../data/DisasterResponse.db')
    engine = create_engine(f'sqlite:///{sys.argv[1]}')
    df = pd.read_sql_table('disaster_table', con = engine)

    X = df['message']
    # All columns except the ones we don't need
    y = df.loc[:, ~df.columns.isin(['id','message', 'original', 'genre'])]

    # Tokenize data
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    def tokenize(text):
        """
        Tokenize text to be used in pipeline
        """
        # normalize case and remove punctuation
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        
        # tokenize text
        tokens = word_tokenize(text)
        
        # lemmatize and remove stop words
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

        return(tokens)

    # Build pipeline

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
                        'tfidf__use_idf': (True, False)
                       ,'clf__estimator__bootstrap': (True, False)
        }

    cv = GridSearchCV(pipeline, parameters, verbose=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train Model

    print("Please wait: model is training\n")

    cv.fit(X_train, y_train)

    # Export pickle file

    # Save to file in the current working directory
    #pkl_filename = "classifier.pkl"
    pkl_filename = f"{sys.argv[2]}"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(cv, file)

    print(f"Success! Model has been exported to {sys.argv[2]}")

if __name__ == '__main__':
    main()