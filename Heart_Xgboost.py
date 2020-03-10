# XGBoost

# Installing XGBoost
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c anaconda py-xgboost

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mission_Prediction_Dataset.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Média das accuracies = ", accuracies.mean())
print("Desvio padrão = ", accuracies.std())

#Applying F1 score
from sklearn.metrics import f1_score
print("F1 score = ", f1_score(y_test, y_pred))
