#===============================================================================================#
# Imports
#===============================================================================================#
import streamlit as sl
import pickle
import Help_text as pre
import re


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = pre._cont_exp(x)
    x = pre._remove_emails(x)
    x = pre._remove_urls(x)
    x = pre._remove_html_tags(x)
    x = pre._remove_rt(x)
    x = pre._remove_accented_chars(x)
    x = pre._remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


#===============================================================================================#

# Functions and Models Prepared

#===============================================================================================#
model = pickle.load(open('Models/best_model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pickle', 'rb'))
X = pickle.load(open('X.pickle','rb'))

# ===============================================================================================#

# Streamlit

# ===============================================================================================#
sl.title("Amazon Product Review Rating Application")

review_summary_text = sl.text_input('Enter Your Review Summary Here')
review_text = sl.text_area('Enter Your Review Here')

if sl.button('Predict'):
    result_review_sum = review_summary_text.title()

    result_review = review_text.title()

    all_review_text = result_review_sum + result_review

    X = get_clean(all_review_text)
    vec = tfidf.transform([X])
    prediction = model.predict(vec)

    sl.success(prediction[0])
