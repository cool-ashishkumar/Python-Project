import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
data= pd.read_csv(r"C:\Users\ASHISH KUMAR\OneDrive\Desktop\Python\Dataset\spam.csv")


data.dropna(inplace = True)

data.duplicated().sum()

data.drop_duplicates(inplace= True)

#training testing

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

X = data.iloc[:, 0]
y= data.iloc[:, 1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(len(X_train))
print(len(X_test))

cv = CountVectorizer()

X_train_v = cv.fit_transform(X_train)
X_test_v = cv.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

lg= LogisticRegression()
l= lg.fit(X_train_v, y_train)
print(l)

y_pred = lg.predict(X_test_v)


conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap="Oranges", fmt="d")

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')


import pickle

pickle.dump(lg, open('emu.sav', 'wb'))

