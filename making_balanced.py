import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from pipeline import pipeline

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

news = pd.read_csv("south_african_news_articles.csv")

random_set = news.sample(frac=1, random_state=42)

# number of elements
num = 850

first_set = random_set.iloc[:num]
# merging with fake news
fake = pd.read_csv("Fakenews.csv", encoding='utf-8',   encoding_errors='replace')

balanced = pd.concat([fake, first_set], axis=0, ignore_index=True)

balanced.drop(["Unnamed: 0"], inplace=True, axis=1)
balanced.dropna(inplace=True)
# making full text

balanced["full_text"] = balanced["title"] + "." + balanced["content"]

balanced["full_text"] = balanced["full_text"].apply(lambda doc: pipeline(doc))

print(balanced["label"].value_counts())

balanced.to_csv('balanced_new.csv', index=False)

