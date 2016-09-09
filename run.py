# -*- coding: utf-8 -*-
"""
Script for document classification task  for Kaggle hate speech detection
Experimenting on different datasets
using different classifiers and feature selection methods
The code also experiments :
- majority vote ensemble over all classifiers
- blending over all classifiers
"""
import argparse
import csv

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
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
from sklearn.ensemble import VotingClassifier


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
                            ngram_range=(1, 3),
                            preprocessor = Document().preprocess,
                            vocabulary = hatebase
                        ),
                    "delta-tfidf": DeltaTfidf(
                            tokenizer = TreebankWordTokenizer().tokenize,
                            preprocessor = Document().preprocess

                    )}

classifiers = {
                "LREG": LogisticRegression(C=1000, penalty="l1", dual=False),
                # "BernoulliNB": BernoulliNB(alpha=.01),
                # "svm_cv": GridSearchCV(
                #     LinearSVC(penalty="l1", dual=False),
                #     [{'C': [0.0001, 0.001, 0.03, 0.1, 1, 3, 10, 100, 1000]}] #range of C coefficients to try
                #     ),
                "LREG_CV": GridSearchCV(
                    LogisticRegression(penalty="l1", dual=False),
                    [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}] #range of C coefficients to try
                    )
                # "svc": GridSearchCV(SVC(probability=True),
                #                     [{'C': [0.001, 0.03, 0.1, 1, 3, 10]}] #range of C coefficients to try
                #     )
                # "ADABOOST-SVM": AdaBoostClassifier(SVC(probability=True))
                # "SGD" : SGDClassifier(loss="hinge", penalty="l1"),
                # "KNN" : KNeighborsClassifier(n_neighbors=5, algorithm='auto')
}

    #Feature Building
features = {
                # "tfidf": FeatureUnion([
                #         ("tfidf", vectorizers["tfidf"])
                # ]),
                # "delta-tfidf": FeatureUnion([
                #         ("delta-tfidf", vectorizers["delta-tfidf"])
                # ]),
                "count": FeatureUnion([
                        ("count", vectorizers["count"])
                ]),
                # "count_dict": FeatureUnion([
                #         ("count_dict", vectorizers["count_dict"])
                # ]),
                # "tfidf-count_dict": FeatureUnion([
                #         ("count_dict", vectorizers["count_dict"]),
                #         ("tfidf", vectorizers["tfidf"])
                # ]),
                # "delta-tfidf-count_dict": FeatureUnion([
                #         ("count_dict", vectorizers["count_dict"]),
                #         ("delta-tfidf", vectorizers["delta-tfidf"])
                # ]),
                "tfidf-count-count_dict": FeatureUnion([
                        ("count", vectorizers["count"]),
                        ("count_dict", vectorizers["count_dict"]),
                        ("tfidf", vectorizers["tfidf"])
                ]),
                # "delta-tfidf-count-count_dict": FeatureUnion([
                #         ("count", vectorizers["count"]),
                #         ("count_dict", vectorizers["count_dict"]),
                #         ("delta-tfidf", vectorizers["delta-tfidf"])
                # ])

    }

fout = open(args.output,"w")
writer = csv.writer(fout)
clfsc = 0

ensemble_estimators = []
for fvector_name,fvector in features.items():
    for clf_name, clf in classifiers.items():
        clfsc += 1

        print "# %s\t%s" % (fvector_name, clf_name)
        pipeline = Pipeline([
            ('features', fvector),
            ('classifier', clf)]
        )

        pipeline.fit(X_train, y_train)
        p = pipeline.predict_proba(X_test)
        p = p[:, 1]

        print classification_report(y_test, pipeline.predict(X_test))
        roc = roc_auc_score(y_test, p)
        print "roc auc is %s" % roc

        # writer.writerow([
        #     fvector_name,
        #     clf_name,
        #     roc
        #     ])

        # adding estimators to the ensemble
        ensemble_estimators.append(("%s-%s" % (fvector_name, clf_name), pipeline))


print "Training the Ensemble.."
ensemble_model = VotingClassifier(ensemble_estimators, voting='soft')
ensemble_model.fit(X_train, y_train)
pred = ensemble_model.predict_proba(X_test)

print classification_report(y_test, ensemble_model.predict(X_test))
roc = roc_auc_score(y_test, pred[:, 1])
print "Total roc auc is %s" % roc

# writer.writerow([
#             "ALL Blending",
#             "ALL Blending",
#             roc
#             ])


fout.close()







