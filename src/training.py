from preprocessing import df
print(df)

#Count Vectorization is a technique used in Natural Language Processing (NLP) to convert text data into numbers by counting how many times each word appears.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message_'])
print(X.toarray())
print(X.shape)

y=df['Category'].values
#Model building using naive bias algorithm

#split data into train data and training data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

#Precision score is the ratio of correctly predicted positive observations to total predicted positives.
precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


import pickle

# save model
pickle.dump(model, open("spam_model.pkl", "wb"))

# save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))



