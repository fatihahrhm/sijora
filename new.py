import pickle
from matplotlib import colors
import pandas as pd

import pandas as pd
import numpy as np
import re, string, sqlite3, tweepy
import matplotlib.pyplot as plt
from emoji import UNICODE_EMOJI

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================================================================== TRAINING ========================================================================================
data_train = pd.read_csv("https://raw.githubusercontent.com/fatihahrhm/sijora/main/dataset/dataset_training_clean.csv")

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

corpus = pd.read_csv("https://raw.githubusercontent.com/fatihahrhm/sijora/main/dataset/corpus.csv")
corpus = list(corpus.iloc[:,0])

vectorizer = TfidfVectorizer(stop_words=stopwords)
tfidf = vectorizer.fit_transform(data_train['Tweet'].values.astype('U'))
tfidf_vec = tfidf.toarray()

X = tfidf_vec
y = data_train.drop(['Tweet'], axis=1).values
sp = SelectPercentile(chi2, percentile=17)
sp_vec = sp.fit_transform(X, y)

X = sp_vec
y = data_train.drop(['Tweet'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=85)

mnb = MultinomialNB(alpha = 0.0090937)
mnb.fit(X_train, y_train.ravel())

# ====================================================================================================================================================================

def crawling(keyword):

    consumer_key = "F1nuhLOaGVleDN9DknxraRXx6"
    consumer_secret = "VGKVWc4YNUlZsD98NcnbsyzvonYcNrWl6cTL8kpjwYXJCvxQgW"
    access_token = "1067941386-ZXPryHL4vhGoVDg7F7m01WqpysZDfsVMHBCAkv5"
    access_token_secret = "4RIeJhw72HVALXX7ttZPR1KID5jixy0YgoArdr8JVpTTq"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    search_words = keyword
    new_search = search_words + " -filter:retweets"

    tweets = tweepy.Cursor(api.search,
                q=new_search,
                lang="id", 
                tweet_mode='extended').items(10)    

    items = []
    for tweet in tweets:
        items.append(tweet.full_text)        

    hasil = pd.DataFrame(items)
    hasil.columns = ['Tweet']
    return hasil

def save_to_csv(data, keyword):
    file_name = keyword + " " + str(date.today()) + ".csv"
    data.to_csv(file_name, index=False)

emoji = pd.read_csv(r'https://raw.githubusercontent.com/fatihahrhm/sijora/main/dataset/emoji.csv', encoding='utf8')

new = []
for i in emoji[' bahagia']:
    new.append(i.strip())

emoji[' bahagia'] = new
emoji = dict(emoji.values)


def is_emoji(s):
    return s in UNICODE_EMOJI

def replace_emoji(tweet):
    result = ' '
    for char in tweet:
        if is_emoji(char):
            result = result + ' ' + emoji.get(char, char)
        else:
            result += emoji.get(char, char)
    return result

# get stopwords
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# initiate stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def data_cleaning(data): 
    cek = []
    for i, text in enumerate (data['Tweet']):
        text = text.lower()
        # stopword removal
        text = stopword.remove(text)
        text = re.sub("\d+", "", text)
        
        # remove @username
        text = re.sub("(@[^\s]+|@[A-Za-z0-9]+)", " ", text)

        # replace emoji
        text = replace_emoji(str(text))        

        # stemming
        text = stemmer.stem(text)
        
        # remove duplicate char
        text = re.sub(r'(.)\1+$', r'\1', text)
        # tokenizing + append
        cek.append(text.strip())

    data['cleantweet'] = cek

    return data

def fe(data):
    tfidf_trasnsformer = vectorizer.transform(data['Tweet'].values.astype('U')).toarray()
    sp_transformer = sp.transform(tfidf_trasnsformer)
    vector = sp_transformer
    
    return vector

def visualize(prediction):
    plt.pie(pd.DataFrame(prediction).value_counts(), labels = ["Positif","Negatif"], startangle=90, autopct='%1.2f%%', 
        colors=['#588b8b','#c8553d'], explode=[0.05,0], shadow=True)
    # plt.bar(pd.DataFrame(prediction).value_counts(), ['Positif', 'Negatif'], color=['#588b8b','#c8553d'])
    plt.savefig("result.png")
    plt.show()

def predict(keyword):
    data = crawling(keyword)
    data = data_cleaning(data)
    vec = fe(data)
    result = mnb.predict(vec)
    
    return result


# =================================================================================================================================================
def prediction_result():
    data = pd.read_json(r"https://sijoraapi.herokuapp.com/sentiment")
    super_dict = {}
    for d in data['Sijora']:
        tmp = dict(d)
        for k, v in tmp.items(): 
            super_dict.setdefault(k, []).append(v)

    inputs = pd.DataFrame(super_dict)['input_mobile']

    for i in inputs:
        hasil = pd.DataFrame(predict(i))    
        count_positive = len(hasil[hasil[0] == 1])
        count_negative = len(hasil[hasil[0] == 0])
        positif = count_positive/len(hasil)*100
        negatif = count_negative/len(hasil)*100

    return {'positif': positif, 'negatif': negatif}

print(prediction_result())


