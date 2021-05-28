#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:32:50 2021

@author: srikanthr
"""
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scoring import *
from utils import *
import os

#%%
datadir=sys.argv[1]
outdir=sys.argv[2]

#%%
symptoms_keys = open(datadir+"/symptoms").readlines()
symptoms_keys =[line.strip() for line in symptoms_keys]

all_data = pd.read_csv(datadir+'/metadata.csv')

#%%
categories = to_dict(datadir+"/category_to_class")
nfolds = open(datadir+'/nfolds').readlines()
nfolds = int(nfolds[0].strip())

#%%
print("Tuning the hyper-parameters")
mslrange= [1,5,10,15,20,25]
avgAUCs={}
for msl in mslrange:
    vAUCs=[];
    for foldId in range(nfolds):

        train_labels = to_dict( datadir+'/fold_'+str(foldId+1)+'/train_labels' )
        for item in train_labels: train_labels[item]=categories[train_labels[item]]
        train_ids = list(train_labels.keys())
        train_data = all_data[all_data.id.isin(train_ids)]
        train_data = train_data[ ["id"]+symptoms_keys ]
        train_data.reset_index()
       
        for key in symptoms_keys:
            train_data[key].fillna(False,inplace=True)
            
        FL = []
        for _,item in train_data.iterrows():
            pid = item['id']
            f = [item[key]*1 for key in symptoms_keys]
            f.append(train_labels[pid])
            FL.append(f)    
        FL = np.array(FL)
        
        #%%
        classifier = DecisionTreeClassifier(criterion='gini',
                                            min_samples_leaf=msl,
                                            class_weight='balanced',
                                            random_state=42)
        classifier.fit(FL[:,:-1],FL[:,-1])
        
        #%%

        val_labels = to_dict(datadir+'/fold_'+str(foldId+1)+'/val_labels')
        for item in val_labels: val_labels[item]=categories[val_labels[item]]
        val_ids = list(val_labels.keys())
        val_data = all_data[all_data.id.isin(val_ids)]
        val_data = val_data[["id"]+symptoms_keys]
        val_data.reset_index()
        
        for key in symptoms_keys:
            val_data[key].fillna(False,inplace=True)
            
        scores={}
        for idx,item in val_data.iterrows():
            pid = item['id']
            f = [item[key]*1 for key in symptoms_keys]
            sc=classifier.predict_proba(np.array(f,ndmin=2))
            scores[pid]=sc[0][1]
            
        with open('temp.txt','w') as f:
            for item in scores: f.write('{} {}\n'.format(item,scores[item]))    
            
        R = scoring(datadir+'/fold_'+str(foldId+1)+'/val_labels', 'temp.txt')
        vAUCs.append(R['AUC'])
        os.remove('temp.txt')
    avgAUCs[msl]=(sum(vAUCs)/len(vAUCs))
#%%
mslbest = max(avgAUCs,key=avgAUCs.get) 
#%%
print("Training on Dev. data")
    
train_labels = to_dict(datadir+'/train_labels')
for item in train_labels: train_labels[item]=categories[train_labels[item]]
train_ids = list(train_labels.keys())
train_data = all_data[all_data.id.isin(train_ids)]
train_data = train_data[["id"]+symptoms_keys]
train_data.reset_index()
    
for key in symptoms_keys:
    train_data[key].fillna(False,inplace=True)
        
FL = []
for idx,item in train_data.iterrows():
    pid = item['id']
    f = [item[key]*1 for key in symptoms_keys]
    f.append(train_labels[pid])
    FL.append(f)    
FL = np.array(FL)
    
#%%
classifier = DecisionTreeClassifier(criterion='gini',
                                    min_samples_leaf=mslbest,
                                    class_weight='balanced', 
                                    random_state=42)
classifier.fit(FL[:,:-1],FL[:,-1])
with open(outdir+'/model.pkl','wb') as f:
    pickle.dump({'scaler':None,'classifier':classifier},f)

#%%
print("test")
test_labels = to_dict(datadir+"/test_labels")
for item in test_labels: test_labels[item]=categories[test_labels[item]]
test_ids = list(test_labels.keys())
test_data = all_data[all_data.id.isin(test_ids)]
test_data = test_data[["id"]+symptoms_keys]
test_data.reset_index()
    
for key in symptoms_keys:
    test_data[key].fillna(False,inplace=True)
        
scores={}
for idx,item in test_data.iterrows():
    pid = item['id']
    f = [item[key]*1 for key in symptoms_keys]
    sc=classifier.predict_proba(np.array(f,ndmin=2))
    scores[pid]=sc[0][1]
        
with open(outdir+'/test_scores.txt','w') as f:
    for item in scores: f.write('{} {}\n'.format(item,scores[item]))    
        
R = scoring(datadir+'/test_labels', outdir+'/test_scores.txt', outdir+'/test_results.pkl')
with open(outdir+"/summary","w") as f:
	f.write("Test AUC {:.3f}\n".format(R['AUC']))
	for se,sp in zip(R['sensitivity'],R['specificity']):
		f.write("Sensitivity: {:.3f}, Specificity: {:.3f}\n".format(se,sp))

with open(outdir+"/feature_importances","w") as f:
	for key,imp in zip(symptoms_keys,classifier.feature_importances_): f.write("{} {}\n".format(key,imp))
