import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import fbeta_score, make_scorer
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import  r2_score

#%%
#Reading Data
df = pd.read_csv('covid_dataset.csv')
df.info()
print(df)

#%%
#null & duplicate finding
print(df.isnull().sum())
print("Duplicate Values: ",df.duplicated().sum())

#%%

X =  df[['Lab Test']]
Y=  df[[('Confirmed case')]]


#%%
x = np.asarray(X,dtype = float)
y = np.asarray(Y,dtype = float)

#%%
from sklearn.linear_model import LinearRegression
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state = 0 )

regressor = LinearRegression() 
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print(y_pred)



#%%
mae = mean_absolute_error(y_train, y_pred)
mse = mean_squared_error(y_train, y_pred)
score1 = metrics.r2_score(y_train, y_pred)

print("Mean Absulate Error:",mae)
print("Mean Squared Error:",mse)
print("R_squer",score1)
#%%
plt.scatter(x_train, y_train, color = 'Orange')

plt.plot(x_train,regressor.predict(x_train),color = 'Green')
plt.title('Lab test vs Confirmed Case')
plt.xlabel('Lab Test')
plt.ylabel('Confirmed Case')
plt.show()

#%%

from sklearn.preprocessing import PolynomialFeatures

tr = PolynomialFeatures(degree = 4 , include_bias = False)

#%%
tr.fit(x)

x_tr = tr.transform(x)

print(x_tr)

model = LinearRegression().fit(x_tr,y)

r_rq = model.score(x_tr, y)

y_prd = model.predict(x_tr)



#%%
score = r2_score(y, y_prd)

print(score)

#%%
mae = mean_absolute_error(y, y_prd)
mse = mean_squared_error(y, y_prd)
score2 = metrics.r2_score(y, y_prd)

print("Mean Absulate Error:",mae)
print("Mean Squared Error:",mse)
print("R_squer",score2)

#%%
plt.scatter(x, y, color = 'green')
plt.plot(x, y_prd, color = 'orange')
plt.title('Lab test vs Confirmed Case')
plt.xlabel('Lab Test')
plt.ylabel('Confirmed Case')
plt.show()
#%%
df.boxplot(column = ['Confirmed case'])

#%%
death = df[['Death Case']]
death.hist(column = ['Death Case'], bins=5)


#%%Lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

lasso = Lasso(alpha=1.0)

lasso.fit(x_train, y_train)

score3 = lasso.score(x_test, y_test)*100 
score =lasso.score(x_train, y_train)*100
print("Accuracy of test dataset",score3,'%')
print("Accuracy of train dataset",score,'%')


#%%Ridge regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

ridge = Ridge(alpha=1.0)

ridge.fit(x_train, y_train)

score4 = ridge.score(x_test, y_test)*100 
score =ridge.score(x_train, y_train)*100
print("Accuracy of test dataset",score4,'%')
print("Accuracy of train dataset",score,'%')


#%%lasticNet regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

elasticnet = ElasticNet(alpha=1.0)

elasticnet.fit(x_train, y_train)

score5 = elasticnet.score(x_test, y_test)*100 
score =elasticnet.score(x_train, y_train)*100
print("Accuracy of test dataset",score5,'%')
print("Accuracy of train dataset",score,'%')



#%% Data Cleaning
df= df.drop(['Day'], axis = 1)
features = df.columns
print(features)
print(df)
features = [x for x in features if x!='Death Case']
print(features)
train,test = train_test_split(df, test_size = 1/3)
print(len(df))
print(len(train))
print(len(test))
#%%
#Logistics Regression
from sklearn.linear_model import LogisticRegression

dt = LogisticRegression(random_state=10)

x_train = train[features]
y_train = train['Death Case']

x_test = test[features]
y_test = test['Death Case']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)


score6 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Logistic regression ", score6,"%")

#%%
#SVM

dt = SVC()

x_train = train[features]
y_train = train['Death Case']

x_test = test[features]
y_test = test['Death Case']

dt = dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
#print(y_pred)

score7 = accuracy_score(y_test, y_pred)*100
print("Accuracy rate with Support Vector Classifer ", score7,"%")

#%%
data = {'Linear':score1, 'Pokynomial':score2, 'Lasso':score3,'Ridge':score4, 
    'Elasticnet':score5,'SVC':score6,'Logistic regression':score7}
name = list(data.keys())
score = list(data.values())
  
fig = plt.figure(figsize = (12, 6))
 
plt.bar(name, score, color ='orange',
        width = 0.4)
 
plt.xlabel("Used Algorithm")
plt.ylabel("Score of different algorithms")
plt.title("Bar chart on different scores of different algorithms")
plt.show()


#%%
#Data Merge
#Reading Data
df = pd.read_csv('covid_dataset.csv')
df.info()


#%%
print(df)


#%%
df1 = pd.read_csv('covid_first_dose.csv')
df1.info()
print(df1)

#%%

A =  df1[['Day']]
B=  df1[['Number of Vaccinations (First Dose)']]



#%%
labelEncoder = preprocessing.LabelEncoder()
for i in A.columns:
   if A[i].dtype == object:
        A[i] = labelEncoder.fit_transform(A[i])
   else:
        pass


#%%
from sklearn.linear_model import LinearRegression
a_train,a_test,b_train,b_test = train_test_split(A,B,test_size=0.5,random_state = 0 )

regressor = LinearRegression() 
regressor.fit(a_train,b_train)

b_pred = regressor.predict(a_test)
print(b_pred)

#%%
plt.scatter(a_train, b_train, color = 'Indigo')
plt.plot(a_train,regressor.predict(a_train),color = 'Red')
plt.title('Date vs Number of Vaccinations')
plt.xlabel('Date')
plt.ylabel('Number of Vaccinations')
plt.show()

#%%
#merged = pd.merge(df, df1, how="outer", on=["Day", "Day"])
#print(merged)
#%%
df2 = pd.read_csv('covid_second_dose.csv')
df2.info()
print(df2)

#%%

C =  df2[['Day']]
D=  df2[['Number of Vaccinations (Second Dose)']]



#%%
labelEncoder = preprocessing.LabelEncoder()
for i in C.columns:
   if C[i].dtype == object:
        C[i] = labelEncoder.fit_transform(C[i])
   else:
        pass


#%%
from sklearn.linear_model import LinearRegression
c_train,c_test,d_train,d_test = train_test_split(C,D,test_size=0.5,random_state = 0 )

regressor = LinearRegression() 
regressor.fit(c_train,d_train)

d_pred = regressor.predict(c_test)
print(d_pred)

#%%
plt.scatter(c_train, d_train, color = 'Blue')

plt.plot(c_train,regressor.predict(c_train),color = 'Black')
plt.title('Date vs Number of Vaccinations')
plt.xlabel('Date')
plt.ylabel('Number of Vaccinations')
plt.show()


#%%
#merged1 = pd.merge(merged, df2, how="outer", on=["Day", "Day"])
#print(merged1)



#%%
#merged1.to_csv(r'E:\9th\CSE303\Projectfinal.csv')