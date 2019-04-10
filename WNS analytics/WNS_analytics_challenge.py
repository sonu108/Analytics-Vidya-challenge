
# coding: utf-8

# In[1]:

import numpy as np
from sklearn import preprocessing, cross_validation , neighbors
import pandas as pd
import pylab as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
#from sklearn.cross_validation import StratifiedKFold
#from sklearn.cross_validation import GridSearchCV
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# In[2]:

df_train = pd.read_csv("train_LZdllcl.csv")
df_test = pd.read_csv("test_2umaH9m.csv")
print df_train.shape,df_test.shape


# In[3]:

df_full = pd.concat([df_train,df_test])


# In[4]:

is_promoted = df_train['is_promoted']
emp_id = df_test['employee_id']


# In[5]:

df_full.drop(['is_promoted'] , inplace=True , axis = 1)
df_full.drop(['employee_id'] , inplace=True , axis = 1)


# In[6]:

#df_full['age'].value_counts()
df_full.columns


# In[7]:

#df_full['avg_training_score'].value_counts()
#df_full.isnull().values.any()
df_full.isnull().sum()


# In[8]:

edu_dummy = pd.get_dummies(df_full['education'] , dummy_na = True, prefix='edu_')
df_full = pd.concat([df_full,edu_dummy] , axis=1 )
df_full.drop(['education'] , axis=1 , inplace=True)


# In[9]:

dpt_dummy = pd.get_dummies(df_full['department'] , prefix='dpt_')
df_full = pd.concat([df_full,dpt_dummy] , axis=1 )
df_full.drop(['department'] , axis=1 , inplace=True)


# In[10]:

df_full['is_male'] = [1 if i=='m' else 0 for i in df_full['gender']]
df_full.drop(['gender'] , inplace=True , axis = 1)


# In[11]:

df_full['previous_year_rating'].fillna(df_full['previous_year_rating'].mean() , inplace = True)


# In[12]:

ch_dummy = pd.get_dummies(df_full['recruitment_channel'] , prefix='r_channel_')
df_full = pd.concat([df_full,ch_dummy] , axis=1 )
df_full.drop(['recruitment_channel'] , axis=1 , inplace=True)


# In[13]:

rg_dummy = pd.get_dummies(df_full['region'] , prefix='region_')
df_full = pd.concat([df_full,rg_dummy] , axis=1 )
df_full.drop(['region'] , axis=1 , inplace=True)


# In[14]:

#df_full.head()
df_full = df_full.astype('float')


# In[15]:

df_train = df_full[:len(df_train)]
df_test = df_full[len(df_train):]


# In[16]:

print df_train.shape , df_test.shape


# In[17]:

X_train , X_test , y_train , y_test = train_test_split(df_train , is_promoted , test_size = 0.3 )
#y_train = y_train.reshape(-1,1)
#y_test = y_test.reshape(-1,1)
print X_train.shape , X_test.shape , y_train.shape , y_test.shape


# In[19]:

#


# In[24]:

gboost = GradientBoostingClassifier(max_depth=5,n_estimators=200)
gboost.fit(df_train,is_promoted)
result = gboost.predict(df_test)


print 'lol'


# In[26]:

#WNS_solution = open('WNS_solution.csv','w')
WNS_solution = pd.DataFrame()
WNS_solution['employee_id'] = emp_id
result = pd.DataFrame(result)
WNS_solution['is_promoted'] = result
WNS_solution.to_csv('WNS_solution4.csv' , index = False)


# In[ ]:

print 'lol'


# In[ ]:



