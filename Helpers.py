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


import re
import os
import sys
import json

import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob
import en_core_web_sm

from sklearn.feature_extraction.text import CountVectorizer

nlp = en_core_web_sm.load()

path = os.path.dirname(os.path.abspath(__file__))
abbreviations_path = os.path.join('Data', 'abbreviations_wordlist.json')


def _get_wordcounts(x):
    length = len(str(x).split())
    return length


def _get_charcounts(x):
    s = x.split()
    x = ''.join(s)
    return len(x)


def _get_avg_wordlength(x):
    count = _get_charcounts(x) / _get_wordcounts(x)
    return count


def _get_stopwords_counts(x):
    l = len([t for t in x.split() if t in stopwords])
    return l


def _get_hashtag_counts(x):
    l = len([t for t in x.split() if t.startswith('#')])
    return l


def _get_mentions_counts(x):
    l = len([t for t in x.split() if t.startswith('@')])
    return l


def _get_digit_counts(x):
    digits = re.findall(r'[0-9,.]+', x)
    return len(digits)


def _get_uppercase_counts(x):
    return len([t for t in x.split() if t.isupper()])


def _cont_exp(x):
    abbreviations = json.load(open(abbreviations_path))

    if type(x) is str:
        for key in abbreviations:
            value = abbreviations[key]
            raw_text = r'\b' + key + r'\b'
            x = re.sub(raw_text, value, x)
        # print(raw_text,value, x)
        return x
    else:
        return x


def _get_emails(x):
    emails = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', x)
    counts = len(emails)

    return counts, emails


def _remove_emails(x):
    return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)


def _get_urls(x):
    urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
    counts = len(urls)

    return counts, urls


def _remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)


def _remove_rt(x):
    return re.sub(r'\brt\b', '', x).strip()


def _remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x


def _remove_html_tags(x):
    return BeautifulSoup(x, 'lxml').get_text().strip()


def _remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x


def _remove_stopwords(x):
    return ' '.join([t for t in x.split() if t not in stopwords])


def _make_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)

    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)


def _get_value_counts(df, col):
    text = ' '.join(df[col])
    text = text.split()
    freq = pd.Series(text).value_counts()
    return freq


def _remove_common_words(x, freq, n=20):
    fn = freq[:n]
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


def _remove_rarewords(x, freq, n=20):
    fn = freq.tail(n)
    x = ' '.join([t for t in x.split() if t not in fn])
    return x


def _remove_dups_char(x):
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


def _spelling_correction(x):
    x = TextBlob(x).correct()
    return x


def _get_basic_features(df):
    if type(df) == pd.core.frame.DataFrame:
        df['char_counts'] = df['text'].apply(lambda x: _get_charcounts(x))
        df['word_counts'] = df['text'].apply(lambda x: _get_wordcounts(x))
        df['avg_wordlength'] = df['text'].apply(lambda x: _get_avg_wordlength(x))
        df['stopwords_counts'] = df['text'].apply(lambda x: _get_stopwords_counts(x))
        df['hashtag_counts'] = df['text'].apply(lambda x: _get_hashtag_counts(x))
        df['mentions_counts'] = df['text'].apply(lambda x: _get_mentions_counts(x))
        df['digits_counts'] = df['text'].apply(lambda x: _get_digit_counts(x))
        df['uppercase_counts'] = df['text'].apply(lambda x: _get_uppercase_counts(x))
    else:
        print('ERROR: This function takes only Pandas DataFrame')

    return df


def _get_ngram(df, col, ngram_range):
    vectorizer = CountVectorizer(ngram_range=(ngram_range, ngram_range))
    vectorizer.fit_transform(df[col])
    ngram = vectorizer.vocabulary_
    ngram = sorted(ngram.items(), key=lambda x: x[1], reverse=True)

    return ngram


