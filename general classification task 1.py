import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
df_train = pd.read_csv("C:/Users/DEBPRIYA/OneDrive/Desktop/Task 1/MOVIE GENRE CLASSIFICATION/train_data.txt",sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
x_test = pd.read_csv("C:/Users/DEBPRIYA/OneDrive/Desktop/Task 1/MOVIE GENRE CLASSIFICATION/test_data.txt",sep=':::', names=['ID', 'TITLE', 'DESCRIPTION'])
df_test_sol= pd.read_csv("C:/Users/DEBPRIYA/OneDrive/Desktop/Task 1/MOVIE GENRE CLASSIFICATION/test_data_solution.txt",sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
df_train.head(3)
x_test.head(3)
df_test_sol
df_train.info()
df_test_sol.info()
import matplotlib.pyplot as plt

genre_counts = df_train['GENRE'].value_counts()

plt.figure(figsize=(20,8))

plt.bar(genre_counts.index, genre_counts.values,color=['red', 'green', 'blue', 'orange', 'purple'])
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)  # Rotate genre labels for better readability
plt.tight_layout()
plt.grid()
plt.show()

most_watched_genre = genre_counts.idxmax()

print("The most watched genre is:", most_watched_genre)
df_train=df_train.drop(columns=['ID'],axis=1)
x_test=x_test.drop(columns=['ID'],axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['GENRE'] = le.fit_transform(df_train['GENRE'])

df_test_sol['GENRE'] = le.fit_transform(df_test_sol['GENRE'])
df_train['combined_text'] = df_train['TITLE'] + ' ' + df_train['DESCRIPTION']
x_test['combined_text'] = x_test['TITLE'] + ' ' + x_test['DESCRIPTION']
X_train=df_train.drop(['GENRE','DESCRIPTION','TITLE'],axis=1)

X_test=x_test.drop(['DESCRIPTION','TITLE'],axis=1)
y_train=df_train['GENRE']
y_test=df_test_sol['GENRE']
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer on X_train
tfidf_vectorizer.fit(X_train['combined_text'])

X_train = tfidf_vectorizer.transform(X_train['combined_text'])
X_test = tfidf_vectorizer.transform(X_test['combined_text'])
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
log_model=LogisticRegression(C=1)
log_model.fit(x_train,y_train)
y_train_pred1=log_model.predict(x_train)
print(classification_report(y_train,y_train_pred1))
