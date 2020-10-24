#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression
# ## Carbon Dioxide Emission of Cars vs. Fuel Consumption
# EDX Machine Learning with Python: A Practical Introduction
# Aaliyah Fiala
# 10/24/2020

# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv


# In[11]:


df = pd.read_csv("FuelConsumptionCo2.csv")


# In[12]:


df.head()


# In[13]:


df.describe()


# In[14]:


cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]


# In[15]:


cdf.head(9)


# In[17]:


viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()


# In[20]:


plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show


# In[21]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[22]:


plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[23]:


plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='pink')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emission")
plt.show()


# In[24]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[25]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color ='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# In[32]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
print('Coefficients:  ', regr.coef_)
print('Intercept: ', regr.intercept_)


# In[34]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='orange')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[36]:


from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_  - test_y)))
print("Residual Sum of Square (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R-squared Score: %.2f" % r2_score(test_y , test_y_))


# In[ ]:




