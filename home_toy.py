#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:11:09 2018

@author: mayritaspring
"""

import os
import pandas as pd
import numpy as np
import warnings
import time
import gc
warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# set path
default_path = "/Users/mayritaspring/Desktop/Github/Home-Credit-Default-Risk/"
os.chdir(default_path)

# read data
application_train = pd.read_csv('../Kaggle data/application_train.csv')

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, label, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if col != label and (df[col].dtype == 'object' or len(df[col].unique().tolist()) < 20)]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    #replace NAs with mean
    df = df.fillna(df.mean())
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns, categorical_columns

# Split to feature and label 
def split_train_test(df, label,key = None, seed = 7, test_size = 0.3):
    from sklearn import cross_validation
    
    #setting
    seed = seed
    test_size = test_size
    
    #give label y
    y = df[label]
    
    #give feature X
    try:
        cols = [col for col in df.columns if col not in [label, key]]
        X = one_hot_encoder(df = df[cols], label = label)[0]
        categorical_columns = one_hot_encoder(df = df[cols], label = label)[2]
    except:
        X = one_hot_encoder(df = df.loc[:, df.columns != label], label = label)[0]
        categorical_columns = one_hot_encoder(df = df.loc[:, df.columns != label], label = label)[2]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test, categorical_columns

# Measure Performance
from  sklearn  import  metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    y_pred = clf.predict(X)  
    if show_accuracy:
        print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred))),"\n"

    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred)),"\n"
        
    if show_confusion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y,y_pred)),"\n"  


#---------------------------------------------#
# lightgbm with Bayesian Optimization
    
# Prepare dataset for Bayesian Optimization    
# use LabelEncoder to convert categorical features to int type before construct Dataset
from sklearn.preprocessing import LabelEncoder
def label_encoder(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """
    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    return input_df, categorical_feats.tolist(), encoder_dict
application_train, categorical_feats, encoder_dict = label_encoder(application_train)
X = application_train.drop('TARGET', axis=1)
y = application_train.TARGET


#Step 1: parameters to be tuned
def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
    params = {'application':'binary','num_iterations':4000, 'learning_rate':0.05, 'early_stopping_round':100, 'metric':'auc'}
    params["num_leaves"] = round(num_leaves)
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = round(max_depth)
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
    return max(cv_result['auc-mean'])



#Step 2: Set the range for each parameter
lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                        'feature_fraction': (0.1, 0.9),
                                        'bagging_fraction': (0.8, 1),
                                        'max_depth': (5, 8.99),
                                        'lambda_l1': (0, 5),
                                        'lambda_l2': (0, 3),
                                        'min_split_gain': (0.001, 0.1),
                                        'min_child_weight': (5, 50)}, random_state=0)


# ### Step 3: Bayesian Optimization: Maximize
# lgbBO.maximize(init_points=init_round, n_iter=opt_round)


# ### Step 4: Get the parameters
# lgbBO.res['max']['max_params']

# ### Put all together
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = categorical_feats, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8.99),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']

opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)

print('Bayesian Optimization Parameters...')
print(opt_params)

####################################################################################
# LightGBM parameters found by Bayesian optimization
# Prepare dataset for Bayesian Optimization 
from sklearn import cross_validation
seed = 7
test_size = 0.3

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

train_data = lgb.Dataset(data = X, label = y, categorical_feature = categorical_feats, free_raw_data=False)


LGBM_bayes = LGBMClassifier(
    nthread=4,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=32,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.04,
    reg_lambda=0.073,
    min_split_gain=0.0222415,
    min_child_weight=40,
    silent=-1,
    verbose=-1)

LGBM_bayes_fit = LGBM_bayes.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
    eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)


measure_performance(X = X_test, y = y_test, clf = LGBM_bayes, show_classification_report=True, show_confusion_matrix=True)

# feature importances
print('Feature importances:', list(LGBM_bayes.feature_importances_))

# visualization
print('Plot feature importances...')
ax = lgb.plot_importance(LGBM_bayes_fit, max_num_features=10)
plt.show()


#####
import seaborn as sns
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

display_importances(feature_importance_df_ = LGBM_bayes.feature_importances_)

#---------------------------------------------#
# lightgbm with grid search

# Prepare dataset for Bayesian Optimization    
# use function split_train_test can help to 1.set label and dataset 2.One-hot encoding
output = split_train_test(df = application_train, label = 'TARGET', key = 'SK_ID_CURR', test_size = 0.3)
X_train, X_test, y_train, y_test, categorical_columns = output[0:5]

# Model
print('Start training...')
estimator = lgb.LGBMClassifier()

param_grid = {
    'objective': ['binary'],  
    'num_leaves': [12,24,36], 
    'learning_rate': [0.01, 0.05, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)

# Final Model
evals_result = {} 
print('Start predicting...')
gbm_final = lgb.LGBMClassifier(objective = gbm.best_params_['objective'],
                              num_leaves = gbm.best_params_['num_leaves'],
                                learning_rate = gbm.best_params_['learning_rate'], 
                              n_estimators = gbm.best_params_['n_estimators'])
gbm_final_fit = gbm_final.fit(X_train, y_train)
measure_performance(X = X_test, y = y_test, clf = gbm_final, show_classification_report=True, show_confusion_matrix=True)


# feature importances
print('Feature importances:', list(gbm_final.feature_importances_))

# visualization
print('Plot feature importances...')
ax = lgb.plot_importance(gbm_final_fit, max_num_features=10)
plt.show()


# Submission file
test_df = pd.read_csv('../Kaggle data/application_test.csv')
out_df = pd.DataFrame({"SK_ID_CURR":test_df["SK_ID_CURR"], "TARGET":gbm_final.predict_proba(test_df.loc[:, test_df.columns != "SK_ID_CURR"])[1]})
out_df.to_csv("submissions_toy.csv", index=False)
