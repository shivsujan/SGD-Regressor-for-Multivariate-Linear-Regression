# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1. Start

step 2. Data Preparation

step 3. Hypothesis Definition

step 4. Cost Function

step 5. Parameter Update Rule

step 6. Iterative Training

step 7. Model Evaluation

step 8. End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SHIV SUJAN S R
RegisterNumber:  212223040194
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("C:/Users/Admin/Desktop/Placement_Data.csv")
data_processed = data.drop(["sl_no", "salary"], axis=1)
le = LabelEncoder()
for col in data_processed.select_dtypes(include=['object']):
  data_processed[col] = le.fit_transform(data_processed[col])
X = data_processed.iloc[:, :-1]
y = data_processed["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
new_data = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]  # Replace with your desired data
predicted_status = model.predict(new_data)
print("Predicted Job Status:", predicted_status[0])
```
## Output:

y_pred :

![Img 1](https://github.com/user-attachments/assets/d488d6bb-0205-4236-a6c5-5f177290852d)

print(classification_report1) :

![Img 2](https://github.com/user-attachments/assets/0437b275-0095-4271-8e5f-164eef3c0ee2)

Ir.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]):

![Img 3](https://github.com/user-attachments/assets/83369ecd-bcd4-4d34-945b-8ad03a4a7936)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
