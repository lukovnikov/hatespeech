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
from classes import DeltaTfidf
from classes.Document import Document


dataset = {"kaggle":"./data/kaggle/",
           "twitter":"./data/twitter/"
           }


parser = argparse.ArgumentParser(description='Sentiment classification Experiments')
parser.add_argument('-d', '--dataset', help='which dataset to run experiment on',required=True)
parser.add_argument('-o','--output', help='output file name', required=True)
args = parser.parse_args()


train = pd.read_csv(dataset[args.dataset.lower()] + "train.csv")
X_train = train['x'].values
y_train = train['x'].values


test = pd.read_csv(dataset[args.dataset.lower()] + "test.csv")
X_test = train['x'].values
y_test = train['x'].values


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
                    # "delta-tfidf": DeltaTfidf(
                    #         tokenizer = TreebankWordTokenizer().tokenize,
                    #         preprocessor = Document().preprocess
                    #     )
}


# kfolds = {
#                 "CV_unBalanced_2C" : create_dataset(dname,
#                     CV = True, neutral = False, balanced = False, n_folds = 5
#                     ),
#                 "CV_unBalanced_3C" : create_dataset(dname,
#                     CV = True, neutral = True, balanced = False, n_folds = 5
#                     ),
#                 "CV_Balanced_2C" : create_dataset(dname,
#                     CV = True, neutral = False, balanced = True, n_folds = 5
#                     ),
#                 "CV_Balanced_3C" : create_dataset(dname,
#                     CV = True, neutral = True, balanced = True, n_folds = 5
#                     ),
#                 "Split_unBalanced_2C" : create_dataset(dname,
#                     CV = False, neutral = False, balanced = False, n_folds = 5
#                     ),
#                 "Split_Balanced_2C" : create_dataset(dname,
#                     CV = False, neutral = False, balanced = True, n_folds = 5
#                     ),
#                 "Split_unBalanced_3C" : create_dataset(dname,
#                     CV = False, neutral = True, balanced = False, n_folds = 5
#                     ),
#                 "Split_Balanced_3C" : create_dataset(dname,
#                     CV = False, neutral = True, balanced = True, n_folds = 5
#                     )
# }


classifiers = {
                # "svm": LinearSVC(penalty="l1", dual=False),
                "svm_cv": GridSearchCV(
                    LinearSVC(penalty="l1", dual=False),
                    [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}] #range of C coefficients to try
                    ),
                "LREG": LogisticRegression(penalty="l1", dual=False),
                # "BernoulliNB" : BernoulliNB(alpha=.01),
                # "SGD" : SGDClassifier(loss="hinge", penalty="l1"),
                # "KNN" : KNeighborsClassifier(n_neighbors=5, algorithm='auto')
}

    #Feature Building
features = {
                # "lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"])]
                #         ),
                # "lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"])]
                #         ),
                "tfidf" : FeatureUnion([
                        ("tfidf", vectorizers["tfidf"])]
                        ),
                # "delta-tfidf" : FeatureUnion([
                #         ("delta-tfidf", vectorizers["delta-tfidf"])]
                #         ),
                "count" : FeatureUnion([
                        ("count", vectorizers["count"])]
                        ),
                # "tfidf_lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"]),
                #         ("tfidf", vectorizers["tfidf"])]
                #         ),
                # "delta-tfidf_lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"]),
                #         ("delta-tfidf", vectorizers["delta-tfidf"])]
                #         ),
                # "tfidf_lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"]),
                #         ("tfidf", vectorizers["tfidf"])]
                #         ),
                # "delta-tfidf_lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"]),
                #         ("delta-tfidf", vectorizers["delta-tfidf"])]
                #         ),
                # "count_lex-domain" : FeatureUnion([
                #         ("lex-domain", vectorizers["lex-domain"]),
                #         ("count", vectorizers["count"])]
                #         ),
                # "count_lex-all" : FeatureUnion([
                #         ("lex-all", vectorizers["lex-all"]),
                #         ("count", vectorizers["count"])]
                #         )
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
        print "roc auc is %s" %  roc_auc_score(y_test, pred)

        writer.writerow([
            fvector_name,
            clf_name
            ])

fout.close()







