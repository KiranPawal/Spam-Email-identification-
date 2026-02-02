import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('E:\Spam Email classifier\Data\spam.csv')
print(df.head())

print(df.info())    # both are string data type
print(df.isnull().sum().sum())    # not null value in dataset

print(df['Category'].value_counts())
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

#preprocessing
print(df.head())

import re
def clean_message(text):
    text = text.lower()                     # lowercase
    text = re.sub(r'[^\w\s]', ' ', text)    # remove punctuation
    text = re.sub(r'\d+', ' ', text)        # remove numbers
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    return text.strip()                     # remove leading/trailing spaces
df['Message'] = df['Message'].apply(clean_message)
print(df.head())


#Remove Stopwords: Words like is, the, and, to don’t help classification.
import nltk
from nltk.corpus import stopwords    #This line imports the stopwords list from NLTK.
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))  #gives a list of English stopwords
def remove_stopwords(text):                   # This defines a function that will remove stopwords from one message.
    words = text.split()
    return ' '.join(word for word in words if word not in stop_words)
df['Message'] = df['Message'].apply(remove_stopwords)
print(df.head())


# Tokenize by splitting on spaces
df['Message_'] = df['Message'].apply(lambda x: x.split())
print(df.head())


#PorterStemmer is an algorithm that reduces words to their root (stem) form by removing suffixes.
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
print(ps.stem('loving'))

df['Message_'] = df['Message_'].apply(lambda words: [ps.stem(word) for word in words])
print(df.head())


# Convert all entries to strings
df['Message_'] = df['Message_'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

# Remove NaNs just in case
df['Message_'] = df['Message_'].fillna('')

#Generate spam WordCloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wc=WordCloud(width=900,height=500,min_font_size=10,background_color="white")
spam_wc=wc.generate(df[df['Category']==1]['Message_'].str.cat(sep=" "))
plt.figure(figsize=(10,6))
plt.imshow(spam_wc,interpolation='bilinear')
plt.show()

