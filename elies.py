# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:06:01 2022

@author: stratos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pandas.plotting as pp
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from datetime import datetime

rskip = 1
usec = [0,1,2,3,4,6]



df = pd.read_excel(r'C:\Users\stratos\Documents\Forecasting_trial\elies\elies.xlsx',skiprows=rskip, usecols= usec)

df.Έτος = pd.to_datetime(df.Έτος) 
df.Έτος = df.Έτος.dt.year 
type(df.Έτος)
df 
pd.DataFrame.to_csv(df, "elies.csv")

#df = df.fillna(0)
df = df.dropna()

print(df)
df.columns = ['Έτος', 'Δέντρα', 'Ελιές', 'Λάδι', 'Απόδοση', 'τιμή κιλού']

plt.bar(df['Έτος'], df['Ελιές'], label = 'ελιές')
plt.bar(df['Έτος'], df['Λάδι'], label = 'λάδι')
plt.plot(df['Έτος'], 1000*df['Απόδοση'], label = '1000*απόδοση', color = "green")
plt.plot(df['Έτος'], 100*df['τιμή κιλού'], label = '100*(τιμή κιλού)', color = "black")
plt.legend()
plt.xlabel("Έτος")
plt.savefig(r"general_analysis.pdf")
plt.show()

df['Ελιές']=pd.to_numeric(df['Ελιές'])
df['Λάδι']=pd.to_numeric(df['Λάδι'])
df['τιμή κιλού']=pd.to_numeric(df['τιμή κιλού'])

print(df)

print(df.corr(method='spearman'))

## Note for mean values delete NA values first.

mean_olive = statistics.mean(df['Ελιές'])
mean_oil = statistics.mean(df['Λάδι'])
mean_ratio = statistics.mean(df['Λάδι'])/statistics.mean(df['Ελιές'])
mean_olive_per_tree = mean_olive/statistics.mean(df['Δέντρα'])
mean_oil_per_tree = mean_oil/statistics.mean(df['Δέντρα'])
print(mean_olive, mean_oil, mean_ratio, mean_olive_per_tree, mean_oil_per_tree)

features = ['Ελιές', 'Λάδι', 'Απόδοση', 'τιμή κιλού']
pp.scatter_matrix(df[features],  diagonal='kde')
plt.savefig(r"correl_olives.pdf")
plt.show()

linear_model = LinearRegression().fit(df['Ελιές'].values.reshape(-1,1),df['Λάδι'])

print(linear_model.coef_,linear_model.intercept_)

score = round(linear_model.score(df['Ελιές'].values.reshape(-1,1),df['Λάδι']),2)
print(score)


def func(t,a,b):
    y = a*t+b
    return y

g = [0.23,+6.73]

c,cov = curve_fit(func,df['Ελιές'],df['Λάδι'],g,maxfev=1000)
funcdata = func(df['Ελιές'],c[0],c[1]) 
plt.plot(df['Ελιές'], funcdata, label="Μοντέλο", color = 'red')
plt.scatter(df['Ελιές'],df['Λάδι'], label = 'Δεδομένα')
plt.xlabel("Ελιές (κιλά)")
plt.ylabel("Λάδι (κιλά)")
plt.title('$R^2=$' +str(score))
plt.legend()
plt.savefig(r"fit_oil_olives.pdf")
plt.show()






