#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:57:07 2021

@author: yoonhakim
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load data
df = pd.read_csv('stats_column_cleaned.csv')

df.columns #check what variables we have

#choose relevant columns

df_model = df[['Pos', 'Tm', 'MP', 'TS%', 'FTr', 'AST%', 'USG%', 'WS', 'VORP']]

# get dummy data; convert categorical variables into dummy variables
df_dum = pd.get_dummies(df_model)

# train test split

from sklearn.model_selection import train_test_split
X = df_dum.drop('WS', axis =1)
y = df_dum.WS.values

# 80% of data used in training set, 20% in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# statsmodel Multiple Linear Regression

import statsmodels.api as sm

X_sm = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary() #Adjusted R-squared is 0.947
# R-squared is 0.948 which means that our model explains 94.8% of fitted data 

# sklearn Multiple Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

# get cross-validation score; default is 5-fold validation
np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error')) #output: -0.5049069039177937

#-----------------------------------------------------------------------------

# Decision Trees

from sklearn.tree import DecisionTreeRegressor
# tuning parameters using GridsearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

dt = DecisionTreeRegressor(random_state = 1)
np.mean(cross_val_score(dt,X_train,y_train, scoring = 'neg_mean_absolute_error')) #-0.6655797235423985

#tuning parameters using GridSearch

dt_param =  {'criterion':('mse','mae'), 
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}

dt_gs = GridSearchCV(dt,dt_param,scoring='neg_mean_absolute_error',
                                 cv = 5, verbose = True, n_jobs = -1)
     
best_dt_gs = dt_gs.fit(X_train,y_train)                             
best_dt_gs.best_score_ #-0.5854267970530014
best_dt_gs.best_estimator_ #DecisionTreeRegressor(max_depth=20, max_features='auto', min_samples_leaf=10, random_state=1)
#-----------------------------------------------------------------------------

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 1)

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error')) #-0.46580067114499457 (lower the better)


# because the total feature space is so large, I used a randomized search
# to narrow down the parameters for the model. I took the best model from this
# and did more grid search

param_grid =  {'n_estimators': [100,500,1000], 
                                  'bootstrap': [True,False],
                                  'max_depth': [3,5,10,20,50,75,100,None],
                                  'max_features': ['auto','sqrt'],
                                  'min_samples_leaf': [1,2,4,10],
                                  'min_samples_split': [2,5,10]}
                                  
clf_rf_rnd = RandomizedSearchCV(rf, param_distributions = param_grid,scoring='neg_mean_absolute_error',
                                n_iter = 100, cv = 5, verbose = True, n_jobs = -1)

best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)

best_clf_rf_rnd.best_params_#{'n_estimators': 500,'min_samples_split': 2,'min_samples_leaf': 1,
                            #'max_features': 'auto','max_depth': 100,'bootstrap': True}
best_clf_rf_rnd.best_score_ #--0.46336251228640035(slightly better than default random forest)
best_clf_rf_rnd.best_estimator_ # RandomForestRegressor(max_depth=100, n_estimators=500, random_state=1)

# Grid Search

rf = RandomForestRegressor(random_state = 1)

parameters =  {'n_estimators': [450,500,550], 
                                  'bootstrap': [True],
                                  'max_depth': [95,100,105],
                                  'max_features': ['auto','sqrt',10],
                                  'min_samples_leaf': [1,2],
                                  'min_samples_split': [2,5,10]}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error', cv = 5, verbose = True, n_jobs = -1)
gs.fit(X_train,y_train)
gs.best_score_ #-0.46336251228640035
gs.best_estimator_ # RandomForestRegressor(max_depth=95, n_estimators=500, random_state=1)

# test models
tpred_lm = lm.predict(X_test)
tpred_dt = best_dt_gs.best_estimator_.predict(X_test)
tpred_rf = best_clf_rf_rnd.best_estimator_.predict(X_test)


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tpred_lm) # 0.5021073044170522
mean_absolute_error(y_test,tpred_dt) # 0.5933855985540939
mean_absolute_error(y_test,tpred_rf) # 0.4519528606965175

# let's predict 2021 win shares using Random Forest Regressor

# 2021 data are that where the player starts with Precious Achiuwa to the last
df.index[df['Player'] == 'Precious Achiuwa']
df_2021 = df_dum[7875:8040]

X_2021 = df_2021.drop('WS', axis =1)
pred_2021 = best_dt_gs.best_estimator_.predict(X_2021)

true_2021 = df[7875:8040]
player_names = true_2021['Player']
ws_2021 = true_2021[['Player','WS']]

pred_2021_player = []
for i, j in zip(pred_2021, player_names):
    pred_2021_player.append({'WS_pred':i,'Player':j})

#top 10 win shares
pred_2021_player = pd.DataFrame(pred_2021_player)
pred_2021_player= pred_2021_player.sort_values([('WS_pred')], ascending = False).reset_index(drop=True).head(10)

#merge with true ws
top_10_player = pred_2021_player.merge(ws_2021, on = 'Player', how = 'inner')

pos = np.arange(len(top_10_player['WS_pred']))

fig, ax = plt.subplots(figsize = (15,8))
plt.bar(pos, top_10_player['WS_pred'], width = 0.4, alpha = 0.75, edgecolor = 'gray', linewidth = 3, label = 'Predicted')
plt.bar(pos + .4, top_10_player['WS'], width = 0.4, alpha = 0.75, edgecolor = 'gray', linewidth = 3, label = 'Actual',color = 'darkseagreen')
for i in pos:
    plt.text(pos[i], 0.5, s = top_10_player['Player'][i],ha='center', va='bottom', rotation = 'vertical',color = 'white', size = 25)

plt.text(x = -1.5, y = 10, s = '2021 NBA Predicted vs Actual Win Shares - Top 10 Players',fontsize = 30, weight = 'bold', alpha = .75)
plt.text(x = -1.5, y = 9, s = 'Wins shares are predicted with Random Forest model',fontsize = 19, alpha = .85)
plt.text(x = -1.5, y = -1.5, s = 'National Basketball Association                                                                                       Source: Basketball-Reference.com ', fontsize = 17,color = '#f0f0f0', backgroundcolor = 'grey')
plt.xticks([],[])
plt.legend(prop={'size': 20})
ax.set_ylabel('Win Shares', size = 25);

#-----------------------------------------------------------------------------
# Productionization
