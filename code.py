import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# import os
# print(os.listdir("../input"))
# import warnings
# warnings.filterwarnings('ignore')


imdb_data=pd.read_csv('D:/Uni/T9/COMP347/Final Project/nlp_project/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)

print("describe: ")
print(imdb_data.describe())

print("sentiment count:", imdb_data['sentiment'].value_counts())

#splitting the dataset into train and test 

train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)

#text normalization

# tokenization of text
tokenizer=ToktokTokenizer()
# setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')
