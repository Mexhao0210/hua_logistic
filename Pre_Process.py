import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
# %matplotlib inline

df = pd.read_csv('train_tweets.txt', '\t', names=['ID', 'Post'])
# print(df.head(10))

def print_plot(index):
    example = df[df.index == index].values[0]
    if len(example) > 0:
        print('ID:',example[0])
        print('Post:', example[1])


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


print_plot(10)
example= clean_text(df.values[0][1])
df.values[0][1] = example
print_plot(10)

X = df.Post
y = df.ID
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


