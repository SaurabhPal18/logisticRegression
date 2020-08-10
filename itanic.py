#one dependent one indpendent linear
#multiple depentent logiical regression
#logistic regression gives answer in yes or no format that is chatagonitical format
#logistic regression is restricted between 0 and 1
#there is a threshhold which gives the decesion wethher the value should be 0 or 1
#.head innprint prints first five value
#missing data shoul dnot be used in data visualization
#hence wranggling is used it deletes that collum that high huge data missing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
data_titanic =pd.read_excel("C:\\Datasheets\\titanic.xlsx")
#print(data_titanic.head)
print("Total number of pasangers travelling: ",str(len(data_titanic)))
sexwise_list=data_titanic['sex'].tolist()
print("Total no.of male passannger",sexwise_list.count('male'))
print("Total no.of female passannger",sexwise_list.count('female'))
fig, ax=plt.subplots(3,2,figsize=(10,10))
sns.countplot(x="survived",data=data_titanic,ax=ax[0][0])
sns.countplot(x="survived",hue='sex',data=data_titanic,ax=ax[0][1])
sns.countplot(x="survived",hue='pclass',data=data_titanic,ax=ax[1][0])
sns.countplot(x="survived",hue='embarked',data=data_titanic,ax=ax[1][1])
fig.show()
data_titanic.info()
print(data_titanic.isnull())
print(data_titanic.isnull().sum())
sns.heatmap(data_titanic.isnull(),yticklabels='false',cmap='viridis',ax=ax[2][0])
data_titanic.drop('body',axis=1,inplace=True)
data_titanic.drop('cabin',axis=1,inplace=True)
data_titanic.drop('boat',axis=1,inplace=True)
data_titanic.drop('home.dest',axis=1,inplace=True)
data_titanic.drop('age',axis=1,inplace=True)
sns.heatmap(data_titanic.isnull(),yticklabels='false',cmap='viridis',ax=ax[2][1])
print(data_titanic.isnull().sum())
sex_categorical=pd.get_dummies(data_titanic['sex'])
print(sex_categorical)
sex_categorical=pd.get_dummies(data_titanic['sex'],drop_first=True)
embarked_categorical=pd.get_dummies(data_titanic['embarked'],drop_first=True)
pclass_categorical=pd.get_dummies(data_titanic['pclass'],drop_first=True)
print(embarked_categorical)
print(sex_categorical)
print(pclass_categorical)
data_titanic=pd.concat([data_titanic,embarked_categorical,pclass_categorical,sex_categorical])
data_titanic.drop(['sex','embarked','name','pclass'],axis=1,inplace=True)
print(data_titanic.head(5))
y=data_titanic['survived']
x=data_titanic.drop(['survived','ticket'],axis=1)
#print(data_titanic.head())
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train=train_test_split(x,y,test_size=0.33,random_state=1)
X_train.fillna(X_train.mean(),inplace=True)
Y_train.fillna(Y_train.mean(),inplace=True)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(Y_test,predictions))
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predictions))
















