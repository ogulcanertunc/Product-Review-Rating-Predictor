import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from yellowbrick.model_selection import feature_importances
from sklearn.metrics import classification_report
import pickle


def score_funct(model, X_train, X_test, y_train, y_test):
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    d = {'Train Accuracy': [acc_train],
         'Validation Accuracy': [acc_test],
         'Accuracy Difference': [acc_train -acc_test]}
    scores_table = pd.DataFrame(data=d)
    return scores_table

df = pd.read_csv("Amazon yorum/preprocessed.csv")

### TF-IDF ###
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,3),analyzer='char')
X = tfidf.fit_transform(df["all_text"])
pickle.dump(tfidf, open("Amazon yorum/tfidf.pickle", "wb"))
pickle.dump(X, open("Amazon yorum/X.pickle", "wb"))
y = df["overall"]

#X.shape
#y.shape

### Train Test Split ###

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Models ###
skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
# -----------------------------------------------------------#

### Decision Tree Classifier ###
dec_tree = DecisionTreeClassifier()
params = {'min_samples_leaf':[3,5,10,15,30,50,100],
          'max_depth':[2,3,4,5,6,7,8,9]}
opt_model = GridSearchCV(dec_tree,
                                 params,
                                 cv=skf,
                                 scoring='accuracy',
                                 return_train_score=True,
                                 n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
dec_tree_model = opt_model.best_estimator_
dec_table = score_funct(dec_tree_model,X_train,X_test, y_train, y_test)
#   Train Accuracy  Validation Accuracy  Accuracy Difference
#0        0.817048             0.801628             0.015421

### Random Forest  ###
rf = RandomForestClassifier()

params = {'min_samples_leaf':[2,3,5,10,15,30,50,100],
          'max_depth':[17,19,21,23,25,27,30]}
opt_model = GridSearchCV(rf,
                                 params,
                                 cv=skf,
                                 scoring='accuracy',
                                 return_train_score=True,
                                 n_jobs=-1)

opt_model.fit(X_train, y_train)
opt_model.best_params_
rf_model = opt_model.best_estimator_
rf_table = score_funct(rf_model,X_train,X_test, y_train, y_test)

#   Train Accuracy  Validation Accuracy  Accuracy Difference
#0        0.842494             0.791455             0.051039


## Logistic Regression ###
logit = LogisticRegression()
params = {'penalty':['l1','l2'],
          'C':[0.2,0.3,0.5,0.7,1,5,10]}
opt_model = GridSearchCV(logit,
                                 params,
                                 cv=skf,
                                 scoring='accuracy',
                                 return_train_score=True,
                                 n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
logit_model = opt_model.best_estimator_
logit_table = score_funct(logit_model,X_train,X_test, y_train, y_test)

#    Train Accuracy  Validation Accuracy  Accuracy Difference
# 0        0.867176             0.812818             0.054358

## Support Vector Machines
svm = SVC()
params = {'kernel':['poly'],
          'degree':[2,3]}
opt_model = GridSearchCV(svm,
                         params,
                         cv=skf,
                         scoring='accuracy',
                         return_train_score=True,
                         n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
svc_model = opt_model.best_estimator_
svc_table_1 = score_funct(svc_model,X_train,X_test, y_train, y_test)
#    Train Accuracy  Validation Accuracy  Accuracy Difference
# 0        0.847837             0.812818             0.035019

params_2 = {'C':[0.25,0.27,0.3,0.32, 0.35,0.4],
            'kernel':['linear'],
            'gamma':['scale','auto']}
opt_model = GridSearchCV(svm,
                         params_2,
                         cv=skf,
                         scoring='accuracy',
                         return_train_score=True,
                         n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
svc_model_2 = opt_model.best_estimator_
svc_table_2 = score_funct(svc_model_2,X_train,X_test, y_train, y_test)
#    Train Accuracy  Validation Accuracy  Accuracy Difference
# 0             0.8             0.790437             0.009563

### KNN ###
knn = KNeighborsClassifier()
params = {'n_neighbors':[5,10,15,20,25,30,40]}
opt_model = GridSearchCV(knn,
                         params,
                         cv=skf,
                         scoring='accuracy',
                         return_train_score=True,
                         n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
knn_model = opt_model.best_estimator_
knn_table = score_funct(knn_model,X_train,X_test, y_train, y_test)
#    Train Accuracy  Validation Accuracy  Accuracy Difference
# 0         0.81145             0.802645             0.008805

### Adaboost ###
ada_boost = AdaBoostClassifier()
params = {'learning_rate':[0.3,0.5, 0.7]}
opt_model = GridSearchCV(ada_boost,
                         params,
                         cv=skf,
                         scoring='accuracy',
                         return_train_score=True,
                         n_jobs=-1)
opt_model.fit(X_train, y_train)
opt_model.best_params_
ada_model = opt_model.best_estimator_
ada_table = score_funct(ada_model,X_train,X_test, y_train, y_test)
#    Train Accuracy  Validation Accuracy  Accuracy Difference
# 0        0.814758             0.803662             0.011096


all_models = pd.concat([dec_table,
                        rf_table,
                        logit_table,
                        svc_table_1,
                        svc_table_2,
                        knn_table,
                        ada_table],
                        axis=0)

all_models

## Saving Model ###
models = [dec_tree_model,rf_model,logit_model,svc_model,svc_model_2,knn_model,ada_model ]

pickle.dump(knn_model, open(f'Amazon yorum/Models/best_model.pkl', 'wb'))







