import json
import pandas as pd 
import numpy as np
import bz2
import pickle
import _pickle as cPickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from config import vect_path, model_path
from util import compressed_pickle


data_path = 'data/shuffled-full-set-hashed.csv'
gs_scoring_critera = 'f1_micro'
models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'param_grid': {
            'penalty': ['elasticnet'],
            'C': np.linspace(0.1, 0.9, 10),
            'l1_ratio': np.linspace(0.15, 0.85, 10),
            'solver': ['saga']
        },
    },
    'MultinomialNB': {
        'model': MultinomialNB(),
        'param_grid': {
            'alpha': np.linspace(0,1,10),
            'fit_prior': [True, False]
        },
    },
}


if __name__ == '__main__':
    # read in, prepare, and split data
    print('\n\n-----------Prepping data-----------\n')
    df = pd.read_csv(data_path, header=None, names=['label', 'content'])
    X = list(df['content'].fillna(''))
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=0.30, random_state=0)


    print('\n\n-----------Fitting Models-----------\n')
    results=[]
    results_df = pd.DataFrame()
    for k in models.keys():
        for vectorizer in [CountVectorizer, TfidfVectorizer]:

            print('\nFitting and testing {}, {}'.format(k, vectorizer))

            # vectorize data 
            vect = vectorizer()
            vect.fit(X_train)
            X_train_vectorized = vect.transform(X_train)

            # instantiate gridsearch
            gcv = GridSearchCV(
                estimator = models[k]['model'],
                param_grid = models[k]['param_grid'],
                scoring=gs_scoring_critera,
                cv=5
            )

            # run gridsearch
            gcv.fit(X_train_vectorized, y_train)

            # store results
            d={}
            d[gs_scoring_critera] = gcv.best_score_
            d['model'] = k
            d['vectorizer'] = vectorizer
            d['params'] = gcv.best_params_

            # if best model so far, save
            if len(results)==0 or gcv.best_score_ > results_df[gs_scoring_critera].max():
                compressed_pickle(vect_path, vect)
                compressed_pickle(model_path, gcv.best_estimator_)

            results.append(d)
            results_df = pd.concat([results_df, pd.DataFrame(results)])
            results_df.to_csv('modeling/model_results.csv', index=False)


    print('\n\n-----------Results-----------\n')
    # print best result
    print('Best result(s):')
    best_result = results_df[results_df[gs_scoring_critera]==results_df[gs_scoring_critera].max()]
    print(best_result)