import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from yellowbrick.model_selection import feature_importances

def classification_funct(model_type, x_train, x_val, y_train, y_val):
    """
    This function is for performing classifcation algorithms such as Logistic Regression, Decision Tree, Random Forest, and SVM.

    Parameters
    ----------
    model_type: 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'
    the type of classifcation algorithm you would like to apply

    x_train: dataframe
    the independant variables of the training data

    x_val: dataframe
    the independant variables of the validation data

    y_train: series
    the target variable of the training data

    y_val: series
    the target variable of the validation data

    """

    model_type = model_type
    x_train = x_train
    y_train = y_train
    x_val = x_val
    y_val = y_val
    scores_table = pd.DataFrame()
    feature_importances = pd.DataFrame()

    if model_type == 'Logistic Regression':
        technique = LogisticRegression(fit_intercept=False)
    elif model_type == 'Decision Tree':
        technique = DecisionTreeClassifier(random_state=42)
    elif model_type == 'Random Forest':
        technique = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=42)
    elif model_type == 'SVM':
        technique = SVC()
    elif model_type == 'Naive Bayes':
        technique = GaussianNB()
    elif model_type == 'KNN':
        technique = KNeighborsClassifier(n_jobs=-1)

def scores(model, x_train, x_val, y_train, y_val):

        """
        Gets the accuracy for the given data and creates a dataframe containing scores.

        Parameters
        ----------
        model: 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'
        the type of classifcation applied

        x_train: dataframe
        the independant variables of the training data

        x_val: dataframe
        the independant variables of the validation data

        y_train: series
        the target variable of the training data

        y_val: series
        the target variable of the validation data

        Returns
        ----------
        scores_table: a dataframe with the model used, the train accuracy and validation accuracy

        """

        acc_train = self.best_model.score(x_train, y_train)
        self.acc_val = self.best_model.score(x_val, y_val)

        d = {'Model Name': [self.model_type],
             'Train Accuracy': [self.acc_train],
             'Validation Accuracy': [self.acc_val],
             'Accuracy Difference': [self.acc_train - self.acc_val]}
        self.scores_table = pd.DataFrame(data=d)

        return self.scores_table
