import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import umap
import plotly.graph_objs as go
import plotly.figure_factory as ff

"""DATA CLEANING"""

# Reading dataset as dataframe
df = pd.read_csv("Labelled.csv")
df2 = df
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window
df['Label'] = df['Label'].map({'Positive':1, 'Negative':0})
print(df)
# Cleaning summaries 
def cleaner(Review):
    soup = BeautifulSoup(Review, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(#|@|http://|https://|www)\S*", " ", souped) # substituting hashtags, @mentions, urls, etc with whitespace
    re2 = re.sub("[^A-Za-z]+"," ", re1) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

"""Creating TFIDF Matrix"""

df['cleaned_Review'] = df.Review.apply(cleaner)
df = df[df['cleaned_Review'].map(len) > 0] # removing rows with cleaned summaries of length 0
print("Printing top 5 rows of dataframe showing original and cleaned summaries....")
print(df[['Review','cleaned_Review']].head())
df['cleaned_Review'] = [" ".join(row) for row in df['cleaned_Review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_Review']
Y = df['Label'] # target column
tfidf = TfidfVectorizer(min_df=.003, ngram_range=(1,1)) # min_df=.003 means that each ngram (unigram, bigram, & trigram) must be present in at least 30 documents for it to be considered as a token (60000*.0005=30). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
print("The created tokens: \n", tfidf.get_feature_names_out())
print("Shape of tfidf matrix: ", data_tfidf.shape)
pd.DataFrame(pd.Series(tfidf.get_feature_names_out())).to_csv('vocabulary_reviews.csv', header=False, index=False)
print("Shape of tfidf matrix: ", data_tfidf.shape)

from pandas.core.algorithms import unique
print(Y.unique())

"""svc"""

print("Implementing SVC.....")
# Implementing Support Vector Classifier
svc_clf = LinearSVC(random_state = 101) # kernel = 'linear' and C = 1

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    svc_clf.fit(X_train, Y_train) # Fitting SVC
    Y_pred = svc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
svc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", svc_mean_accuracy)

print("Implementing NBC.....")
# Implementing Naive Bayes Classifier
nbc_clf = MultinomialNB()

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    nbc_clf.fit(X_train, Y_train) # Fitting NBC
    Y_pred = nbc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
nbc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", nbc_mean_accuracy)

import joblib
classifier = LinearSVC().fit(data_tfidf,Y)
joblib.dump(classifier,'svc.sav')

"""**Deployment**"""

from imblearn.over_sampling import SMOTE
model =  joblib.load('svc.sav')
vocabulary = pd.read_csv('vocabulary_reviews.csv', header=None)
vocabulary_dict = {}
for i, word in enumerate(vocabulary[0]):                              
      vocabulary_dict[word] = i
print(vocabulary_dict)
tfidf = TfidfVectorizer(vocabulary = vocabulary_dict,lowercase=False)

# Reading new data as dataframe
# Reading new data as dataframe
df = pd.read_csv("unlabelled.csv")
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window

# Cleaning reviews
def cleaner(Review):
    soup = BeautifulSoup(Review, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", souped) # substituting @mentions, urls, etc with whitespace
    re2 = re.sub("[^A-Za-z]+"," ", re1) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]   
    return lemmas

df['cleaned_review'] = df.Review.apply(cleaner)
df = df[df['cleaned_review'].map(len) > 0] # removing rows with cleaned tweets of length 0
print("Printing top 5 rows of dataframe showing original and cleaned tweets....")
print(df[['Review','cleaned_review']].head())
df['cleaned_review'] = [" ".join(row) for row in df['cleaned_review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_review']
print(data.shape)
tfidf.fit(data)
data_tfidf1 = tfidf.transform(data)
y_pred = model.predict(data_tfidf1)
print(len(y_pred.reshape(-1,1)))
print(df.shape)
#### Saving predicted ratings to csv
df['predicted_rating'] = y_pred.reshape(-1,1)
df.to_csv('predicted_rating.csv', index=False)

"""umap visualisation"""

# Implementing UMAP to visualize dataset
u = umap.UMAP(n_neighbors=150, min_dist=0.1,random_state=101)
x_umap = u.fit_transform(data_tfidf)

op = list(df2['Label'])
review = list(df2['Review'])

data_ = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=df2['Label'], colorscale='rdylbu', opacity=0.5),
                                text=[f'Label: {a}<br>review: {b}' for a,b in list(zip(op, review))],
                                hoverinfo='text')]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 650, height = 550,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data_, layout=layout)
fig.show()