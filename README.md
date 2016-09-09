# Detecting Insults in Social Commentary Code Base 

code base to Predict whether a comment posted during a public discussion is considered insulting to one of the participants.

## Datasets : 

- Kaggle (2k): https://www.kaggle.com/c/detecting-insults-in-social-commentary
- Twitter dataset (16K) : 


## models used : 

Ensemble of classifiers : Logistic regression, SVM, Kmeans Classifiers, 
Multiple bag of words features such as :   TFIDF, DeltaTFIDF, Count, Dictionary 
Ensemble Classifier using :
-- Majority vote
-- Blending 


Best Acheved score : 0.862 roc_auc on kaggledataset and 0.93 on twitterdataset  
