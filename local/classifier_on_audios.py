#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 23:11:43 2021

@author: srikanthr
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os, pickle, sys
from scoring import *
from utils import *

#%%
sound_category=sys.argv[1]
classifier = sys.argv[2]
datadir=sys.argv[3]
featsfil = sys.argv[4]
outdir=sys.argv[5]  

#%%
cRange = [i for i in range(-7,8)]
categories = to_dict(datadir+"/category_to_class")
nfolds = open(datadir+'/nfolds').readlines()
nfolds = int(nfolds[0].strip())

if not os.path.exists(outdir):
	os.mkdir(outdir)

features_data=pd.read_csv(featsfil)
#%%
test_labels = to_dict(datadir+"/test_labels")
for item in test_labels: test_labels[item]=categories[test_labels[item]]
test_features_data = features_data[ features_data.file_name.isin(list(test_labels.keys()))]
test_features_data = test_features_data.to_numpy()
test_features = {}
for item in test_features_data:
    test_features[item[-1]]=item[1:-1]
del test_features_data
#%%
if classifier == 'lr':
    from sklearn.linear_model import LogisticRegression as clf
elif classifier in ['linearSVM','rbfSVM']:
    from sklearn.svm import SVC as clf
else:
    raise ValueError("Unknown classifier")
#%%
averageValidationAUCs={}
print("=========== Tuning hyperparameters")
for c in cRange:
    C = 10**c
     
    outfolder_root = "{}/results_{}_c{}".format(outdir,classifier,c)
    if not os.path.exists(outfolder_root):
        os.mkdir(outfolder_root)

    #%%
    valAUCs=[]
    for fold in range(nfolds):

        train_labels = to_dict(datadir+'/fold_'+str(fold+1)+'/train_labels')
        for item in train_labels: train_labels[item]=categories[train_labels[item]]

        train_features_data = features_data[ features_data.file_name.isin(list(train_labels.keys()))]
        train_features_data = train_features_data.to_numpy()
        train_features = {}
        for item in train_features_data:
            train_features[item[-1]]=item[1:-1]
        del train_features_data
    
        train_F=[]
        train_l=[]
        for item in train_labels:
            train_l.append(train_labels[item])
            train_F.append(train_features[item])
        train_F=np.array(train_F)
        train_l=np.array(train_l)
        #%%
        scaler = StandardScaler()
        scaler.fit(train_F)
        if classifier=='lr':
            model = clf( C=C,
                        penalty='l2',
                        class_weight='balanced',
                        solver='liblinear',
                        random_state=np.random.RandomState(42))
        elif classifier=='linearSVM':
            model = clf( C=C,
                        kernel='linear',
                        class_weight='balanced',
                        probability=True,
                        random_state=np.random.RandomState(42))
        elif classifier=='rbfSVM':
            model = clf( C=C,
                        kernel='rbf',
                        class_weight='balanced',
                        probability=True,
                        random_state=np.random.RandomState(42))

        model.fit(scaler.transform(train_F),train_l)

        if not os.path.exists(outfolder_root+'/fold_'+str(fold+1)):
            os.mkdir(outfolder_root+'/fold_'+str(fold+1))

        pickle.dump({'classifier':model,'scaler':scaler},
                    open(outfolder_root+'/fold_'+str(fold+1)+'/model.pkl','wb'))

        #%%    
        val_labels = to_dict(datadir+'/fold_'+str(fold+1)+'/val_labels')
        for item in val_labels: val_labels[item]=categories[val_labels[item]]    
    
        val_features_data = features_data[ features_data.file_name.isin(list(val_labels.keys()))]
        val_features_data = val_features_data.to_numpy()
        val_features = {}
        for item in val_features_data:
            val_features[item[-1]]=item[1:-1]
        del val_features_data
        
        scores={}
        for item in val_labels:
            feature = val_features[item]
            feature=feature.reshape(1,-1)
            scores[item] = model.predict_proba(scaler.transform(feature))[0][1]
        

        with open(outfolder_root+'/fold_'+str(fold+1)+'/val_scores.txt','w') as f:
            for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
        scores = scoring(datadir+'/fold_'+str(fold+1)+'/val_labels',
                         outfolder_root+'/fold_'+str(fold+1)+'/val_scores.txt',
                         outfolder_root+'/fold_'+str(fold+1)+'/val_results.pkl')
        valAUCs.append(scores['AUC'])

        scores={}
        for item in test_labels:
            feature = test_features[item]
            feature=feature.reshape(1,-1)
            scores[item] = model.predict_proba(scaler.transform(feature))[0][1]

        with open(outfolder_root+'/fold_'+str(fold+1)+'/test_scores.txt','w') as f:
            for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
        scores = scoring(datadir+'/test_labels',
                         outfolder_root+'/fold_'+str(fold+1)+'/test_scores.txt',
                         outfolder_root+'/fold_'+str(fold+1)+'/test_results.pkl')
        del scaler,model
    averageValidationAUCs[c]=sum(valAUCs)/len(valAUCs)
#%%
best_c = max(averageValidationAUCs,key=averageValidationAUCs.get)
bestC = 10**best_c
print("Best Val. AUC {} for C={}".format(averageValidationAUCs[best_c],bestC))
#%%

outfolder_root = "{}/results_{}".format(outdir,classifier)
if not os.path.exists(outfolder_root):
    os.mkdir(outfolder_root)
with open(outfolder_root+"/bestC","w") as f: 
    f.write("{}".format(bestC))
    
train_labels = to_dict(datadir+"/train_labels")
for item in train_labels: train_labels[item]=categories[train_labels[item]]

train_features_data = features_data[ features_data.file_name.isin(list(train_labels.keys()))]
train_features_data = train_features_data.to_numpy()
train_features = {}
for item in train_features_data: train_features[item[-1]]=item[1:-1]    

train_F=[]
train_l=[]
for item in train_labels:
    train_l.append(train_labels[item])
    train_F.append(train_features[item])
train_F=np.array(train_F)
train_l=np.array(train_l)
#%%    
scaler = StandardScaler()
scaler.fit(train_F)
    
if classifier=='lr':
    model = clf( C=bestC,
                penalty='l2',
                class_weight='balanced',
                solver='liblinear',
                random_state=np.random.RandomState(42))
elif classifier=='linearSVM':
    model = clf( C=bestC,
                kernel='linear',
                class_weight='balanced',
                probability=True,
                random_state=np.random.RandomState(42))
elif classifier=='rbfSVM':
    model = clf( C=bestC,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=np.random.RandomState(42))

model.fit(scaler.transform(train_F),train_l)

pickle.dump({'classifier':model,'scaler':scaler},
            open(outfolder_root+'/model.pkl','wb'))


scores={}
for item in test_labels:
    feature = test_features[item]
    feature=feature.reshape(1,-1)
    scores[item] = model.predict_proba(scaler.transform(feature))[0][1]

with open(outfolder_root+'/test_scores.txt','w') as f:
    for item in scores: f.write("{} {}\n".format(item,scores[item]))
    
scores = scoring(datadir+'/test_labels',
                 outfolder_root+'/test_scores.txt',
                 outfolder_root+'/test_results.pkl')
with open(outfolder_root+"/summary","w") as f:
	f.write("Test AUC {:.3f}\n".format(scores['AUC']))
	for se,sp in zip(scores['sensitivity'],scores['specificity']):
		f.write("Sensitivity: {:.3f}, Specificity: {:.3f}\n".format(se,sp))
        
print("Test AUC: {}".format(scores['AUC']))