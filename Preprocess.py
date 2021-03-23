import pandas as pd
import numpy as np
pd.set_option("max_columns", None)


df_= pd.read_csv("Data/df_sub.csv")
df_.dropna(inplace=True)
df_.head()
df=pd.DataFrame()
df["all_text"] = df_["summary"]+df_["reviewText"]
df["overall"] =df_["overall"]

df["overall"].value_counts()

import preprocess_folder as pre
import re


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = pre.cont_exp(x)
    x = pre.remove_emails(x)
    x = pre.remove_urls(x)
    x = pre.remove_html_tags(x)
    x = pre.remove_rt(x)
    x = pre.remove_accented_chars(x)
    x = pre.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


df["all_text"] = df["all_text"].apply(lambda x: get_clean(x))
#df.head()

df.to_csv('Amazon yorum/preprocessed.csv',index=False)