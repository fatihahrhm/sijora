import pandas as pd
import numpy as np
from datetime import date, timedelta
import re, string, sqlite3, tweepy

def crawling(keyword):
    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    search_words = keyword
    new_search = search_words + " -filter:retweets"

    tweets = tweepy.Cursor(api.search,
                q=new_search,
                lang="id", 
                tweet_mode='extended').items(2000)    

    items = []
    tweetId = []
    dates = []
    username = []
    for tweet in tweets:
        items.append(tweet.full_text)
        tweetId.append(tweet.id)
        dates.append(tweet.created_at)
        username.append(tweet.user.screen_name)

    hasil = pd.DataFrame(list(zip(tweetId, dates, username, items)), columns=['tweetId','date','username','tweet'])

    return hasil
 
def save_to_csv(data, keyword):
    file_name = keyword + " " + str(date.today()) + ".csv"
    data.to_csv(file_name, index=False)

keyword = "larangan mudik"
tweets = crawling(keyword)
save_to_csv(tweets, keyword)
print(tweets)
