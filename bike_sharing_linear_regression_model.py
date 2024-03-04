#!/usr/bin/env python
# coding: utf-8

# ##### Import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


# ##### Steps to follow:
# 1. Reading and understanding the data
# 2. Preparing data for modeling (split into train & test)
# 3. Training the model
# 4. Residual analysis
# 5. Prediction and evaluation on the test data

# # 1. Reading and understanding the data

# In[2]:


day_data = pd.read_csv("data.csv")
day_data.head()


# In[3]:


day_data.shape


# In[4]:


day_data.info()


# In[5]:


print("season: ",day_data['season'].unique())
print("weathersit", day_data['weathersit'].unique())


# In[6]:


day_data.describe()


# In[7]:


#convering to categorical variable.
day_data['weathersit'] = day_data['weathersit'].replace({1: 'Clear', 2: 'Mist', 3: 'Light_Snow', 4: 'Heavy_Rain'})
day_data['season'] = day_data['season'].replace({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})
day_data.head()


# - Dropping the unnecessary columns

# In[8]:


# Droping the instant column as it just a record index, and not useful for the model
day_data = day_data.drop(columns=['instant'])

# Dropping the casual and registered columns as cnt column is a addition of both
day_data = day_data.drop(columns=['casual', 'registered'])

# since dteday column is having the same data as month so we can drop the dteday column as well
day_data = day_data.drop(columns=['dteday'])

day_data.head()


# - Visualising the data

# In[9]:


sns.set(style="ticks")

numerical_columns = ['cnt', 'temp', 'atemp', 'hum', 'windspeed']

pairplot = sns.pairplot(day_data[numerical_columns], diag_kind='kde', markers='o', palette='deep')
pairplot.fig.suptitle('Pairplot of Numerical variables', y=1.02, fontsize=16)
pairplot.tight_layout()
plt.show()


# In[10]:


#From the above graph, it is clear that the 'atemp' and 'temp' columns are highly correlated;
correlation = day_data['temp'].corr(day_data['atemp'])
print("Correlation of temp and atemp columns: ", correlation)


# In[11]:


#The correlation of temp and atemp columns is almost 1 hence we can remove any of them, here we will remove the atemp column.
day_data = day_data.drop(columns=['atemp'])
day_data.head()


# In[12]:


# Set Seaborn style
sns.set(style="whitegrid")

# Visualizing the categorical columns
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))

sns.boxenplot(x='season', y='cnt', data=day_data, ax=axes[0, 0], palette="deep")
axes[0, 0].set_title('Season')

sns.boxenplot(x='yr', y='cnt', data=day_data, ax=axes[0, 1], palette="deep")
axes[0, 1].set_title('Year')

sns.boxenplot(x='mnth', y='cnt', data=day_data, ax=axes[0, 2], palette="deep")
axes[0, 2].set_title('Month')

sns.boxenplot(x='holiday', y='cnt', data=day_data, ax=axes[1, 0], palette="deep")
axes[1, 0].set_title('Holiday')

sns.boxenplot(x='weekday', y='cnt', data=day_data, ax=axes[1, 1], palette="deep")
axes[1, 1].set_title('Weekday')

sns.boxenplot(x='workingday', y='cnt', data=day_data, ax=axes[1, 2], palette="deep")
axes[1, 2].set_title('Working Day')

sns.boxenplot(x='weathersit', y='cnt', data=day_data, ax=axes[2, 0], palette="deep")
axes[2, 0].set_title('Weather Situation')

# # Adjust layout
fig.tight_layout()

# Display the plots
plt.show()


# # 2. Preparing the data for the modeling

# In[13]:


#Creating dummy variables
season_column = pd.get_dummies(day_data['season'], drop_first=True)
month_column = pd.get_dummies(day_data['mnth'], drop_first=True)
weekday_column = pd.get_dummies(day_data['weekday'], drop_first=True)
weathersit_column = pd.get_dummies(day_data['weathersit'], drop_first=True)


# In[14]:


#Concating the dummy variables with the original data and renaming the columns for better interpretation
day_data = pd.concat([day_data, season_column, weathersit_column], axis=1)

day_data = pd.concat([day_data, month_column], axis=1)
day_data.rename(columns={2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}, inplace=True)

day_data = pd.concat([day_data, weekday_column], axis=1)
day_data.rename(columns={1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}, inplace=True)

day_data.head()


# In[15]:


#After concatinating the dummy variables we can remove the original categorical columns
day_data = day_data.drop(columns=['season', 'mnth', 'weekday', 'weathersit'], axis=1)
day_data.head()


# In[16]:


day_data.shape


# - Splitting data into train and test

# In[17]:


# Rearranging the columns for better understanding before splitting the data into train and test
day_data = day_data.reindex(columns=['cnt', 'yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'spring', 'summer', 'winter', 'Light_Snow', 
                                     'Mist', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
day_data.head()


# In[18]:


train_data, test_data = train_test_split(day_data, train_size=0.7, random_state=100)
print(train_data.shape)
print(test_data.shape)


# - Rescalling the features

# In[19]:


# Initiate the object
scaler = MinMaxScaler()

#Fit on data
train_data[['temp', 'hum', 'windspeed', 'cnt']] = scaler.fit_transform(train_data[['temp', 'hum', 'windspeed', 'cnt']])
train_data.head()


# # 3. Traininig the model

# In[20]:


#Creating a heatmap to check the correlation of the variables with count variable
plt.figure(figsize = (30, 20))
sns.heatmap(train_data.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[21]:


train_data.head()


# - Creating X_train and y_train

# In[22]:


y_train = train_data.pop('cnt')
X_train = train_data


# In[23]:


y_train.head()


# In[24]:


X_train.shape


# - Building the model using statsmodel
# - We will build the model with all the variables and remove one variable at a time based on the significance.

# In[25]:


# Checking the columns
day_data.columns


# #### Bulding the model with all the variables

# - We will go with the RFE approach to select the best 15 features for the model

# In[26]:


lm_rfe = LinearRegression()

# fit the model
lm_rfe.fit(X_train, y_train)

# running the RFE to select the 15 features
lm_model_rfe = RFE(estimator=lm_rfe, n_features_to_select=15)
lm_model_rfe = lm_model_rfe.fit(X_train, y_train)


# In[27]:


list(zip(X_train.columns,lm_model_rfe.support_,lm_model_rfe.ranking_))


# In[28]:


# Printing the RFE supported columns
rfe_supported_cols = X_train.columns[lm_model_rfe.support_]
rfe_supported_cols


# In[29]:


# Printing the RFE rejected columns
X_train.columns[~lm_model_rfe.support_]


# In[30]:


# Creating X_train dataframe with RFE selected variables
X_train = X_train[rfe_supported_cols]
X_train.head()


# In[31]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# - Creating a function to check VIF

# In[32]:


# Function to check VIF
def check_vif(df):
    vif = pd.DataFrame()
    X = df
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# - We will remove the column having VIF >5

# In[33]:


check_vif(X_train)


# - Since the VIF of the hum is >5, we will remove it

# In[34]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['hum'], axis=1)
X_train.head()


# In[35]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[36]:


check_vif(X_train)


# - Since the p value of the 'May' variable is high and VIF is is low, we will remove it

# In[37]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['May'], axis=1)
X_train.head()


# In[38]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[39]:


check_vif(X_train)


# - Here the VIF of the 'temp' variable is high but we have seen earlier in the pair plot that 'temp' variable is corelated with the target variable.
# - So it is an important variable for model building, hence we will remove the 'Oct' variable and check if the VIF of the variable is descresing or not.

# In[40]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['Oct'], axis=1)
X_train.head()


# In[41]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[42]:


check_vif(X_train)


# - The p value of the 'spring' is high compared to other variables, so we will remove it and check if the VIF is descresing or not

# In[43]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['spring'], axis=1)
X_train.head()


# In[44]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[45]:


check_vif(X_train)


# - The p value of the 'Mar' is high compared to other variables, so we will remove it and check if the VIF is descresing or not

# In[46]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['Mar'], axis=1)
X_train.head()


# In[47]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[48]:


check_vif(X_train)


# - The p value of the 'Aug' is high compared to other variables, so we will remove it and check if the VIF is descresing or not

# In[49]:


# Removing the variables having high VIF
X_train = X_train.drop(columns=['Aug'], axis=1)
X_train.head()


# In[50]:


# Adding the constant
X_train_model = sm.add_constant(X_train)

# Creating the model
lr_sm = sm.OLS(y_train, X_train_model)

# fit the model
lr_sm_model = lr_sm.fit()

#parameters of the model
lr_sm_model.summary()


# In[51]:


check_vif(X_train)


# - As we can see that after removing the 'Aug' variable, The VIF of the other variables is descreased significantly.
# - The model appears to be good and it is showing minimal multicollinearity among predictors and statistically significant p-values for all features.
# - For now we will consider it as a our final model

# # 4. Residual analysis

# In[52]:


y_train_pred = lr_sm_model.predict(X_train_model)
y_train_pred


# In[53]:


residuals = y_train - y_train_pred
sns.displot(residuals)


# # 5. Prediction and evaluation on the test data

# In[54]:


#Fit on data
test_data[['temp', 'hum', 'windspeed', 'cnt']] = scaler.transform(test_data[['temp', 'hum', 'windspeed', 'cnt']])
test_data.head()


# In[55]:


test_data.describe()


# In[56]:


y_test = test_data.pop('cnt')
X_test = test_data


# - Adding a constant to make predictions

# In[57]:


X_test_sm_model = sm.add_constant(X_test)
X_test_sm_model.head()


# - As we removed some of the variables from training data set during the model building, hence we will remove those variables from test data.

# In[58]:


X_test_sm_model = X_test_sm_model.drop(columns=['hum', 'May', 'Oct', 'spring', 'Mar', 'Aug', 'workingday', 'Feb', 'Apr', 'Jun', 'Jul', 
                                                 'Nov', 'Dec', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], axis=1)
X_test_sm_model.head()


# - Making the prediction

# In[59]:


y_test_pred = lr_sm_model.predict(X_test_sm_model)


# - Evaluating the model

# In[60]:


print("R-squared score on test data:", r2_score(y_true=y_test, y_pred=y_test_pred))


# - Creating a graph to evaluate the model

# In[61]:


fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot
ax.scatter(y_test, y_test_pred, alpha=0.8, edgecolors='k', s=80)

# Identity line (y=x)
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)

ax.set_title('Actual vs Predicted', fontsize=15)
ax.set_xlabel('Actual Values (y_test)', fontsize=13)
ax.set_ylabel('Predicted Values (y_test_pred)', fontsize=13)

