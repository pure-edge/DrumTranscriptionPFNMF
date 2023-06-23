import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

#df = pd.read_csv('timbre.csv')  # Read the CSV file into a DataFrame
#df = df.drop_duplicates()
#df.to_csv('timbre.csv', index=False)

df = pd.read_csv('timbre.csv')
print(df.info())
print(df.head())
class_counts = df['technique'].value_counts()  # Count the occurrences of each class label
print(class_counts)

X = df.drop('technique', axis=1)  # Features
y = df['technique']  # Labels

# apply min-max scaling to features
scaler = MinMaxScaler() 
X = scaler.fit_transform(X)

model = SVC(kernel='rbf')

cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Create empty lists to store the scores for each fold
precision_scores = []
recall_scores = []
f1_scores = []

# Perform cross-validation
for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision_scores.append(precision_score(y_test, y_pred, average='micro'))
    recall_scores.append(recall_score(y_test, y_pred, average='micro'))
    f1_scores.append(f1_score(y_test, y_pred, average='micro'))

avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

print("Average Micro-Precision:", avg_precision)
print("Average Micro-Recall:", avg_recall)
print("Average Micro F1-score:", avg_f1)


precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

avg_precision = precision_scores.mean()
avg_recall = recall_scores.mean()
avg_f1 = f1_scores.mean()

print("\nAverage Macro-Precision:", avg_precision)
print("Average Macro-Recall:", avg_recall)
print("Average Macro-F1-score:", avg_f1)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)