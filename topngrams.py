# -*- coding: utf-8 -*-
import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from classes.Document import Document
from sklearn.cross_validation import train_test_split
from unidecode import unidecode

parser = argparse.ArgumentParser(description='Feature Selection Experiments')
parser.add_argument('-d', '--dataset', help='which dataset to run experiment on', required=False)
parser.add_argument('-o', '--output',  help='ouput ""directory"" name', required=True)
args = parser.parse_args()

########################
### LOADING DATASETS ###
########################
dataset = {"kaggle":"./data/kaggle/",
           "twitter":"./data/twitter/"
           }

if args.dataset.lower() == "kaggle":
    train = pd.read_csv(dataset[args.dataset.lower()] + "train.csv")
    X_train = train['x'].values
    y_train = train['y'].values

    test = pd.read_csv(dataset[args.dataset.lower()] + "test.csv")
    X_test = test['x'].values
    y_test = test['y'].values

elif args.dataset.lower() == "all":

    train = pd.read_csv(dataset['kaggle'] + "train.csv")
    X_train1 = train['x'].values
    y_train1 = train['y'].values

    test = pd.read_csv(dataset['kaggle'] + "test.csv")
    X_test = test['x'].values
    y_test = test['y'].values

    data = pd.read_csv(dataset['twitter'] + "train.csv")
    X_train2 = np.array([unidecode(x) for x in data['x'].values])
    y_train2 = np.array([int(i) - 1 for i in data['y'].values])

    X_train = np.concatenate([X_train1, X_train2])
    y_train = np.concatenate([y_train1, y_train2])

else:
    data = pd.read_csv(dataset[args.dataset.lower()] + "train.csv")
    X = np.array([unidecode(x) for x in data['x'].values])
    y = np.array([int(i) - 1 for i in data['y'].values])

    ids = range(0, len(X))
    train, test = train_test_split(ids, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

hatebase = pd.read_csv('./data/hatebase/hatebase.csv').vocabulary.values


vectorizer = CountVectorizer(
        tokenizer=TreebankWordTokenizer().tokenize,
        ngram_range=(1, 5),
        preprocessor = Document().preprocess)
classifier = GridSearchCV(
                    LogisticRegression(penalty="l1", dual=False),
                    [{'C': [0.0001, 0.001, 0.1, 1, 10, 100]}] #range of C coefficients to try
                    )

X = vectorizer.fit_transform(X_train)
classifier.fit(X, y_train)

coeffs = classifier.best_estimator_.coef_
fn = [i.encode("utf-8") for i in vectorizer.get_feature_names()]

sfn = np.array(fn, dtype=np.str_)[coeffs.nonzero()[1]]
sfv = np.array(coeffs[coeffs.nonzero()],dtype=np.float32)
selected_features = np.array([sfn, sfv])

# sorting feature vector according to coefficient values
selected_features = selected_features[:, sfv.argsort()]

f = open(args.output, 'w')
for i, j in zip(selected_features[0],selected_features[1]):
    f.write("%s\t%s\n" % (i, j))
f.close()
