#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:33:47 2023

@author: stratos
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



rskip = 1
usec = [0,1,2,3,4,6]



df = pd.read_excel(r'/Users/stratos/Documents/Python_projects/Learning_examples/elies/elies.xlsx',skiprows=rskip, usecols= usec)

#df = df.fillna(0)
df = df.dropna()

print(df)
df.columns = ['Έτος', 'Δέντρα', 'Ελιές', 'Λάδι', 'Απόδοση', 'τιμή κιλού']

plt.plot(df['Έτος'], df['Ελιές'], label = 'ελιές')
plt.show()

per_drop = np.log(1+df['Ελιές']).pct_change()

mu, sigma = per_drop.mean(), per_drop.std()

print(per_drop)

df = df.set_index('Έτος')
df.index = pd.to_datetime(df.index)

up_date = '2016-01-01'
#target = 'Λάδι'
target = 'Ελιές'

ind = df[df.index==up_date]
in_value = ind[target][0]
print(in_value)

df = df[df.index > up_date]
print(df)
no_pred = 7
df = df.reset_index()
df

for _ in range(100):
    sim_rets = np.random.normal(mu,sigma,no_pred)
    sim_value =  in_value*(sim_rets+1)
    #plt.axhline(in_value, color='black', ls='--')
    plt.title("Monte Carlo forecasting")
    plt.plot(sim_value, linewidth=0.8)
    plt.plot(df.index, df[target], color = 'black')
    plt.xlabel('Years ahead from 2017')
    plt.ylabel(target)
    
plt.show()


