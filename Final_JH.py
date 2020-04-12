# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:17:28 2020

@author: akash
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier
  
jh = pd.read_csv("train_2.csv")
jh_test = pd.read_csv("test_2.csv")
jh.head()
jh_test.head()
jh.shape
jh_test.shape
jh.isnull().sum()
jh_test.isnull().sum()
jh.describe()
jh.columns

jh_y = jh['gender']
fea = ['Session_ID','Duration', 'products']
jh_X = jh[fea]
jh_X.columns
jh_X_test = jh_test[fea]
jh_X_test.shape

jh_train_X, jh_val_X, jh_train_y, jh_val_y = train_test_split(jh_X, jh_y, test_size = 0.2,random_state=42)
jh_train_X
jh_train_y
jh_val_X
jh_val_y

jh_xgb = XGBClassifier(learning_rate =0.2,n_estimators=20000,max_depth=5,min_child_weight=2,
 gamma=0.2,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,
 scale_pos_weight=1,seed=27)
jh_xgb.fit(jh_train_X,jh_train_y)
jh_xgb_pred = jh_xgb.predict(jh_val_X)
jh_xgb_pred  
confusion_matrix(jh_val_y,jh_xgb_pred)
accuracy_score(jh_val_y,jh_xgb_pred)
pred = jh_xgb.predict(jh_X_test)
pred
pred_xgb = pd.DataFrame(pred)
pred_xgb.to_csv("Predict.csv")

param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch1.fit(jh_train_X,jh_train_y)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch2.fit(jh_train_X,jh_train_y)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_

param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch3.fit(jh_train_X,jh_train_y)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

param_test4 = {'max_features':range(0,3,1)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15, min_samples_split=1000, min_samples_leaf=30, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch4.fit(jh_train_X,jh_train_y)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=15,min_samples_split=1000, min_samples_leaf=30, subsample=0.8, random_state=10,max_features=2),
param_grid = param_test5, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch5.fit(jh_train_X,jh_train_y)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_

jh_gb = GradientBoostingClassifier(learning_rate=0.005, n_estimators=2000,max_depth=15,
  min_samples_split=1000,min_samples_leaf=30, subsample=0.85, random_state=10, max_features=2)
jh_gb.fit(jh_train_X,jh_train_y)
jh_gb_pred = jh_xgb.predict(jh_val_X)
jh_gb_pred  
confusion_matrix(jh_val_y,jh_gb_pred)
accuracy_score(jh_val_y,jh_gb_pred)
pred = jh_gb.predict(jh_X_test)
pred
pred_gb = pd.DataFrame(pred)
pred_gb.to_csv("Predict.csv")