#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:19:54 2018

@author: canf
"""

import pandas as pd

from sklearn import ensemble
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import gzip
import xgboost as xgb

def loadData(df, scaler=None):
    data = pd.DataFrame(index=range(len(df)))
    
    data = df.get(['X','Y'])
    
    DayOfWeeks = df.DayOfWeek.unique()
    DayOfWeekMap = {}
    i = 0
    for day in DayOfWeeks:
        DayOfWeekMap[day] = i
        i += 1
    data = data.join(df['DayOfWeek'].map(DayOfWeekMap))
    
    PdDistricts = df.PdDistrict.unique()
    PdDistrictMap = {}
    i = 0
    for s in PdDistricts:
        PdDistrictMap[s] = i
        i += 1
    data = data.join(df['PdDistrict'].map(PdDistrictMap))
        
    date_time = pd.to_datetime(df.Dates)
    year = date_time.dt.year
    data['Year'] = year
    month = date_time.dt.month
    data['Month'] = month
    day = date_time.dt.day
    data['Day'] = day
    hour = date_time.dt.hour
    data['hour'] = hour
    minute = date_time.dt.minute
    time = hour*60+minute
    data['Time'] = time
    
    data['StreetCorner'] = df['Address'].str.contains('/').map(int)
    data['Block'] = df['Address'].str.contains('Block').map(int)
    
    X = data.values
    Y = None
    if 'Category' in df.columns:
        Y = df.Category.values
    
    return X, Y, scaler
    
def RFpredict(X,Y,Xhat):
    clf = ensemble.RandomForestClassifier()
    clf.set_params(min_samples_split=1000)
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf

def NBpredict_Gauss(X,Y,Xhat):
    clf = GaussianNB()
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf
    
def NBpredict_Bernoulli(X,Y,Xhat):
    clf = BernoulliNB()
    clf.fit(X,Y)
    Yhat = clf.predict_proba(Xhat)
    return Yhat,clf

train = pd.read_csv("./input/train.csv")
X,Y,scaler = loadData(train)

test = pd.read_csv("./input/test.csv")
Xhat,_,__ = loadData(test)

print(X.shape)
print(Y.shape)

input("Press Enter to continue...")

#Predictions

#dtrain = xgb.DMatrix(X, label=Y)
#param = {'max_depth' : 5, 'eta' : 0.01,  'objective': 'binary:logistic', \
#         'subsample' : 0.9}
#num_round = 10
#bst = xgb.train(param, dtrain, num_round)
#dtest = xgb.DMatrix(Xhat)
#Yhat_bst = bst.predict(dtest)

Yhat,clf = RFpredict(X,Y,Xhat)

print(Yhat.shape)
print(Yhat)

input("Press Enter to continue...")

submission = pd.DataFrame(Yhat,columns=clf.classes_)
submission['Id'] = test.Id.tolist()
submission.to_csv(gzip.open('RF.csv.gz','wt'),index=False)

Yhat2,clf2 = NBpredict_Gauss(X,Y,Xhat)

submission2 = pd.DataFrame(Yhat2,columns=clf2.classes_)
submission2['Id'] = test.Id.tolist()
submission2.to_csv(gzip.open('NB.csv.gz','wt'),index=False)

Yhat3,clf3 = NBpredict_Bernoulli(X,Y,Xhat)

submission3 = pd.DataFrame(Yhat3,columns=clf3.classes_)
submission3['Id'] = test.Id.tolist()
submission3.to_csv(gzip.open('Bern.csv.gz','wt'),index=False)