import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

messages = pd.read_csv('smsspamcollection\SMSSpamCollection', sep='\t', names=['label', 'message'])

wordnet = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

multi = MultinomialNB()
spam_detection_model = multi.fit(X_train, y_train)

y_pred = spam_detection_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)
print('Model accuracy is:\t', acc)
cm = confusion_matrix(y_test, y_pred)
print(cm)