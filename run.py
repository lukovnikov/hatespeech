# -*- coding: utf-8 -*-

import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.grid_search import GridSearchCV
from classes.DeltaTfidf import DeltaTfidf
from classes.Document import Document
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from unidecode import unidecode


dataset = {"kaggle":"./data/kaggle/",
           "twitter":"./data/twitter/"
           }

parser = argparse.ArgumentParser(description='Sentiment classification Experiments')
parser.add_argument('-d', '--dataset', help='which dataset to run experiment on',required=True)
parser.add_argument('-o','--output', help='output file name', required=True)
args = parser.parse_args()


########################
### LOADING DATASETS ###
########################

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


vectorizers = {
                    "tfidf": TfidfVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1, 5), norm="l1",
                            preprocessor = Document().preprocess
                        ),
                    "count": CountVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1, 5),
                            preprocessor = Document().preprocess
                        ),
                    "count_dict": CountVectorizer(
                            tokenizer=TreebankWordTokenizer().tokenize,
                            ngram_range=(1, 5),
                            preprocessor = Document().preprocess,
                            vocabulary = hatebase
                        ),
                    "delta-tfidf": DeltaTfidf(
                            tokenizer = TreebankWordTokenizer().tokenize,
                            preprocessor = Document().preprocess
                        )
}


classifiers = {
                "LREG": LogisticRegression(penalty="l1", dual=False),
                "BernoulliNB" : BernoulliNB(alpha=.01),
                "svm_cv": GridSearchCV(
                    LinearSVC(penalty="l1", dual=False),
                    [{'C': [0.0001, 0.001, 0.03, 0.1, 1, 3, 10, 100, 1000]}] #range of C coefficients to try
                    )
                # "SGD" : SGDClassifier(loss="hinge", penalty="l1"),
                # "KNN" : KNeighborsClassifier(n_neighbors=5, algorithm='auto')
}

    #Feature Building
features = {
                "tfidf": FeatureUnion([
                        ("tfidf", vectorizers["tfidf"])
                ]),
                "delta-tfidf": FeatureUnion([
                        ("delta-tfidf", vectorizers["delta-tfidf"])
                ]),
                "count": FeatureUnion([
                        ("count", vectorizers["count"])
                ]),
                "count_dict": FeatureUnion([
                        ("count_dict", vectorizers["count_dict"])
                ]),
                "tfidf-count-count_dict": FeatureUnion([
                        ("count", vectorizers["count"]),
                        ("count_dict", vectorizers["count_dict"]),
                        ("tfidf", vectorizers["tfidf"])
                ]),
                "delta-tfidf-count-count_dict": FeatureUnion([
                        ("count", vectorizers["count"]),
                        ("count_dict", vectorizers["count_dict"]),
                        ("delta-tfidf", vectorizers["delta-tfidf"])
                ])

    }

fout = open(args.output,"w")
writer = csv.writer(fout)

for fvector_name,fvector in features.items():
    for clf_name, clf in classifiers.items():

        print "# %s\t%s" % (fvector_name, clf_name)

        pipeline = Pipeline([
                        ('features', fvector),
                        # ('select',selector),
                        ('classifier', clf)])


        pipeline.fit(X_train,y_train)
        pred = pipeline.predict(X_test)

        #metrics of each class
        m1 = np.array(precision_recall_fscore_support(y_test, pred))
        print classification_report(y_test, pred)
        roc = roc_auc_score(y_test, pred)
        print "roc auc is %s" %  roc

        writer.writerow([
            fvector_name,
            clf_name,
            roc
            ])

fout.close()







