import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2

df = pd.read_csv("https://raw.githubusercontent.com/fatihahrhm/sijora/main/dataset/dataset_training_clean.csv")

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

corpus = pd.read_csv("https://raw.githubusercontent.com/fatihahrhm/sijora/main/dataset/corpus.csv")
corpus = list(corpus.iloc[:,0])

vectorizer = TfidfVectorizer(stop_words=stopwords, vocabulary=corpus)
tfidf = vectorizer.fit_transform(df['Tweet'].values.astype('U'))
vector = tfidf.toarray()

X = vector
y = df.drop(['Tweet'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=85)

mnb = MultinomialNB(alpha = 0.0090937)
mnb.fit(X_train, y_train.ravel())
