# %%
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

normalize = lambda document: document.lower()

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', string)

def remove_unwanted(document):

    # remove user mentions
    document = re.sub("@[A-Za-z0-9_]+"," ", document)
    # remove URLS
    document = re.sub(r'http\S+', ' ', document)
    # remove hashtags
    document = re.sub("#[A-Za-z0-9_]+","", document)
    # remove emoji's
    document = remove_emoji(document)
    # remove punctuation
    document = re.sub("[^0-9A-Za-z ]", "" , document)
    # remove double spaces
    document = document.replace('  ',"")
    
    return document.strip()

def remove_words(tokens):
    stopwords = nltk.corpus.stopwords.words('english') # also supports german, spanish, portuguese, and others!
    stopwords = [remove_unwanted(word) for word in stopwords] # remove puntcuation from stopwords
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens


lemma = WordNetLemmatizer()

def lemmatize(tokens):
    lemmatized_tokens = [lemma.lemmatize(token, pos = 'v') for token in tokens]
    return lemmatized_tokens


stem = PorterStemmer()

def stemmer(tokens):
    stemmed_tokens = [stem.stem(token) for token in tokens]
    return stemmed_tokens


def pipeline(document, rule = 'lemmatize'):
    # first lets normalize the document
    document = normalize(document)
    # now lets remove unwanted characters
    document = remove_unwanted(document)
    # create tokens
    tokens = document.split()
    # remove unwanted words
    tokens = remove_words(tokens)
    # lemmatize or stem or
    if rule == 'lemmatize':
        tokens = lemmatize(tokens)
    elif rule == 'stem':
        tokens = stemmer(tokens)
    else:
        print(f"{rule} Is an invalid rule. Choices are 'lemmatize' and 'stem'")
    
    return " ".join(tokens)

