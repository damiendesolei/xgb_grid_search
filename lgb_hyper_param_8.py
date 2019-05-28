# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:44:06 2019

@author: ZRJ
"""
import time
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import NuSVR

import warnings
warnings.filterwarnings('ignore')


#read in feature martix
X_tr = pd.read_csv(r'C:/Users/ZRJ/Desktop/LANL-Earthquake-Prediction/features/X_train.csv')
y_tr = pd.read_csv(r'C:/Users/ZRJ/Desktop/LANL-Earthquake-Prediction/features/y_train.csv')
X_sub = pd.read_csv(r'C:/Users/ZRJ/Desktop/LANL-Earthquake-Prediction/features/X_submit.csv')
X_sub = X_sub.drop(columns=['seg_id'])

scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)  
X_sub_scaled = pd.DataFrame(scaler.transform(X_sub), columns=X_sub.columns) 


#prepare training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_tr, test_size=0.20, random_state=14)


#define leaning rate shrinkage
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


#setup lgb parameters
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'mae', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': -1,
            'categorical_feature': 'auto',
            }

#starting parameters
param_test ={
# =============================================================================
#              'num_leaves': sp_randint(100, 1000),
#              'max_depth': sp_randint(1, 10),
#              'min_data_in_leaf': sp_randint(1, 100),
# =============================================================================
# =============================================================================
#              'min_child_samples': sp_randint(100, 1000), 
#              'min_child_weight': sp_uniform(loc=0, scale=1.0),#[1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#              'subsample': sp_uniform(loc=0.2, scale=0.8), 
#              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6)
# =============================================================================
             'bagging_freq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'bagging_fraction':sp_uniform(loc=0.0, scale=1.0),
             'reg_alpha': sp_uniform(loc=0.0, scale=1.0),#[0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': sp_uniform(loc=0.0, scale=1.0)#[0, 1e-1, 1, 5, 10, 20, 50, 100]
             }


n_hyper_parameter_points_to_test = 100

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 50000 define only the absolute maximum
lgb_reg = lgb.LGBMRegressor(random_state=14, silent=True, metric='gamma', boosting='gbdt', n_jobs=1, n_estimators=50000, bagging_seed=1,
                            max_depth=4, min_data_in_leaf=2, num_leaves=321, colsample_bytree=0.5155872558124424, min_child_samples=815,
                            min_child_weight=0.5122614044196322, subsample=0.5555279793433687
                           )
gs = RandomizedSearchCV(estimator=lgb_reg, param_distributions=param_test, n_iter=n_hyper_parameter_points_to_test,
                                 scoring='neg_mean_absolute_error', cv=5, refit=True, random_state=14, verbose=True, n_jobs = 6)

gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))





#steup cv fold
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)

#master model
def train_model(X=X_train_scaled, X_test=X_sub_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = 5)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction



#lgb
lgb_params = {'max_depth': 4, 'min_data_in_leaf': 2, 'num_leaves': 321,
              'colsample_bytree': 0.5155872558124424, 'min_child_samples': 815, 'min_child_weight': 0.5122614044196322, 'subsample': 0.5555279793433687,
              'bagging_fraction': 0.7782997733917691, 'bagging_freq': 1, 'reg_alpha': 0.06902141857731026, 'reg_lambda': 0.506358866207801
              }
oof_lgb, prediction_lgb, feature_importance = train_model(params=lgb_params, model_type='lgb', plot_feature_importance=True)




submission = pd.read_csv(r'C:/Users/ZRJ/Desktop/LANL-Earthquake-Prediction/sample_submission.csv')
submission['time_to_failure'] = prediction_lgb
# submission['time_to_failure'] = prediction_lgb_stack
print(submission.head())
submission.to_csv('C:/Users/ZRJ/Desktop/LANL-Earthquake-Prediction/submission/27May2019/submission.csv',index=False)