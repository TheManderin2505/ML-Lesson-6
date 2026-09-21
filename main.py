import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


data = pd.read_csv("student-mat.csv")

print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


list1 = ["school","sex","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]

for i in list1:
    data[i] = encoder.fit_transform(data[i])
    
print(data.info())
print(data.head())

l2 = ["famrel","freetime","goout","Dalc","Walc","health","absences"]
for j in l2:
    x = list1.append(j)

print(list1)
#x = data[["school", "sex", "age", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]]
x = data[list1]
y = data["G3"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn.ensemble import RandomForestClassifier
#n_estimators = 100 - num of tress
classifier = RandomForestClassifier(n_estimators=100)


classifier.fit(xtrain,ytrain)
ypredict=classifier.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
acuracy = accuracy_score(ytest,ypredict)
acuracy = round(acuracy*100,2)
print("Acuracy is : ",acuracy,"%")

matrix = confusion_matrix(ytest,ypredict)
sb.heatmap(matrix,annot=True,fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Accuracy")

plt.show()
