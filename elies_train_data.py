#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:48:43 2022

@author: stratos
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pickle

rskip = 1
usec = [0,1,2,3,4,6]



df = pd.read_excel(r'/Users/stratos/Documents/Python_projects/Learning_examples/elies/elies.xlsx',skiprows=rskip, usecols= usec)
df = pd.DataFrame(df)
#df = df.fillna(0)
df = df.dropna()

#print(df)
df.columns = ['Έτος', 'Δέντρα', 'Ελιές', 'Λάδι', 'Απόδοση', 'τιμή κιλού']
df = df[['Ελιές', 'Λάδι']]


df['Ελιές']=pd.to_numeric(df['Ελιές'])
df['Λάδι']=pd.to_numeric(df['Λάδι'])
#df['τιμή κιλού']=pd.to_numeric(df['τιμή κιλού'])

print(df)


predict = "Λάδι"

X = np.array(df.drop([predict],axis=1))
Y = np.array(df[predict])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.9)#, random_state=1)
#print(X_test, Y_test)

#train the model over 10000 iterations
"""best = 0
for _ in range(10000):
    #train the model with 60% of the data
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1)#, random_state=1)
    
    linear = linear_model.LinearRegression()
    
    #predict a model from the trained data  
    linear.fit(X_train,Y_train)
    
    #get the accuracy of the model
    acc = linear.score(X_test,Y_test)
    print(acc)
    if acc > best:
         best = round(acc,2) 

    with open("linear_model.pickle", "wb") as f: 
        pickle.dump(linear,f)"""

Pred_values = X_test


pickle_in = open("linear_model.pickle","rb")
linear = pickle.load(pickle_in)
best = round(linear.score(X_test,Y_test),2)
pred_dummy = linear.predict(Pred_values)
D = []

#artificial olive kg data
future_list = [[3250],[2033],[5348],[3716],[5887],[2200],[5282],[6120],[7758],[5230],[2825],
[3558],[2368],[1718],[6901],[7304],[3316],[6500],[7500],[5420],[2000],[3242],[4223],[7122]]

#future predictions
pred_future = linear.predict(future_list)
print("future list is:\n", pred_future)

for y in range(len(pred_dummy)):
    print(pred_dummy[y],X_test[y], Y_test[y])
    D.append((pred_dummy[y], Y_test[y]))

cols = ['prediction', 'Λάδι',]
result = pd.DataFrame(D, columns=cols) 
"""result_2 = pd.DataFrame(C, columns="artificial") 
print(result_2)"""

result = result.reset_index()
plt.scatter(result['index'],result['prediction'], label="ML prediction")
plt.scatter(result['index'],result['Λάδι'], label="data")
plt.ylabel("Κιλά")
plt.xlabel("Χρονιά")
plt.legend()
#plt.title("End points of batteries")
plt.title("Προβλέψεις,   " +'$R^2=$' +str(best))
plt.savefig(r"ML_oil_prediction.pdf")
plt.show()


