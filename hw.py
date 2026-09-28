import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import datasets

data1 = datasets.load_wine()
print(data1.keys())

data = pd.DataFrame(data1.data)
data.columns = data1.feature_names

data["alcohol"] = data1.target

print(data.head())
print(data.info())

y = data["alcohol"]
data.drop("alcohol",axis=1)
x=data

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size= 0.2, random_state=5)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

model.fit(xtrain,ytrain)
ypredict=model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
acuracy =  accuracy_score(ytest,ypredict)
acuracy = round(acuracy*100,2)

print("acuracy = ",acuracy,"%")

matrix = confusion_matrix(ytest,ypredict)
sb.heatmap(matrix,annot=True,fmt="d")

plt.title("confusion_matrix")
plt.xlabel("prediction")
plt.ylabel("acuracy")

plt.show()