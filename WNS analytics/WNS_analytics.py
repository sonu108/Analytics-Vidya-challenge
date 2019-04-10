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
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

df_train = pd.read_csv("train_LZdllcl.csv")
#df_test = pd.read_csv("test_2umaH9m.csv")

print df_train.head()




