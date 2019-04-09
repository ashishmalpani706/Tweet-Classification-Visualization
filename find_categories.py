# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import math
import pandas as pd
import numpy as np
import re
import nltk

#Load dataset
df = pd.read_csv('main.csv')#,nrows = 1500)

df['original_text'] = df['text']

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt  

# remove twitter handles (@user)
df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")

#TO DO - REMOVE ASCII EXTENDED
for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    tweet = tweet.lower()
    tweet = re.sub(r"\S+\/.\S+ *\S+|.\S+html|\S+-\S+|\d*\/\d+|\d+|\S+%\S+|\S+:\S*|\S+=\S+|.#\S+", "", tweet)
    tweet = tweet.encode("ascii", errors="ignore").decode()
    df.iloc[i,df.columns.get_loc('text')] = tweet
	
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english'))
for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    word_tokens = word_tokenize(tweet) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    tweet = ' '.join(filtered_sentence)
    df.iloc[i,df.columns.get_loc('text')] = tweet
	
custom_words = ['via','rt','fav','…','am','et','pm','n\'t','y\'all']
for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    querywords = tweet.split()
    resultwords  = [word for word in querywords if word.lower() not in custom_words]
    result = ' '.join(resultwords)
    df.iloc[i,df.columns.get_loc('text')] = result
	
import string
remove = string.punctuation + ".‘’\''“”°…-—––•・®.:#"
for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    tweet = ' '.join(word.strip(remove) for word in tweet.split())
    tweet = tweet.strip()
    df.iloc[i,df.columns.get_loc('text')] = tweet

# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    tweet =([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(tweet)])
    tweet = ' '.join(tweet)
    df.iloc[i,df.columns.get_loc('text')] = tweet
	
# REVIEW NEEDED
words = set(nltk.corpus.words.words())
for i in range(len(df)):
    tweet = df.iloc[i,df.columns.get_loc('text')]
    tweet = ' '.join(word for word in tweet.split() if len(word)>3)
    df.iloc[i,df.columns.get_loc('text')] = tweet
	

#----------END--OF--PRE-PROCESSING----------#

train_df = pd.read_csv('C:\\Ashish\\Project\\dataset\\clean-trainset1.csv')#,nrows = 1500)

from sklearn.model_selection import train_test_split
import seaborn as sns 

def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=33)
 
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

trial = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'),
                             min_df=5,norm='l2')),
    ('classifier', MultinomialNB(alpha=0.30)),
])
 
train(trial, train_df.text, train_df.category)

#---------------Classifier trained---------------

predicted_categories = trial.predict(df.text)

df['category'] = pd.Series(predicted_categories)

df.to_csv('C:\\Ashish\\Project\\dataset\\r2v-data.csv')