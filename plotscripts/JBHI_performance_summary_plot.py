#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:11:15 2021

@author: srikanthr and neeraj
"""

import pickle
import numpy as np

rootpath='../'
savefig=True
plotsdir='plots'
#%%
def to_dict(fil,cat_to_dict=None):
    data = open(fil).readlines()
    odata={}
    for line in data:
        key,val=line.strip().split()
        try:
            odata[key]=float(val)
        except:
            if cat_to_dict:
                odata[key]=cat_to_dict[val]
            else:
                odata[key]=val
    return odata
#%%
sthreshold=0.95

best_val_auc_models={}
for category in ['breathing','cough','speech']:
    best_val_auc_models[category]=category+'/results_lr' 
best_val_auc_models['symptoms']='symptoms'
labels=to_dict(rootpath+'data/test_labels',{'p':1,'n':0})


out_data={}
for item in best_val_auc_models:
    resdir = rootpath+best_val_auc_models[item]
    R = pickle.load(open(resdir+'/test_results.pkl','rb'))
    inds=np.where(1-R['FPR']>=sthreshold)[0]    
    threshold = R['thresholds'][inds[0]]
    scores = to_dict(resdir+'/test_scores.txt')
    labels=to_dict(rootpath+'data/test_labels',{'p':1,'n':0})
    sys_out = {}
    for i in scores: 
        if scores[i]>=threshold: 
            sys_out[i]=1
        else: 
            sys_out[i]=0
    tps=0;tns=0
    for i in labels:
        if sys_out[i]==labels[i]:
            if labels[i]==0:
                tns+=1
            if labels[i]==1:
               tps+=1 
    positives = sum(labels.values())
    negatives = len(labels)-positives
    sensitivity = tps/positives
    specificity = tns/negatives
    spositives = sum(sys_out.values())
    snegatives = len(sys_out)-spositives
    PPV = tps/spositives
    NPV = tns/snegatives if snegatives>0 else 0
    out_data[item.capitalize()+" ("+item[:2].upper()+")"] = [R['AUC'],sensitivity,specificity,PPV,NPV]    
    print("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(item,R['AUC'],sensitivity,specificity,PPV,NPV))
    
#%%%
for item in ['BrCo','BrSp','CoSp','BrCoSp','BrSy','CoSy','SpSy','BrCoSy','BrSpSy','CoSpSy']:

    R = pickle.load(open(rootpath+'fusion/'+item+'_results.pkl','rb'))
    inds=np.where(1-R['FPR']>=sthreshold)[0]    
    threshold = R['thresholds'][inds[0]]
    scores = to_dict(rootpath+'fusion/'+item+'_scores.txt')
    labels=to_dict(rootpath+'data/test_labels',{'p':1,'n':0})
    sys_out = {}
    for i in scores: 
        if scores[i]>threshold: 
            sys_out[i]=1
        else: 
            sys_out[i]=0
    tps=0;tns=0
    for i in labels:
        if sys_out[i]==labels[i]:
            if labels[i]==0:
                tns+=1
            if labels[i]==1:
               tps+=1 
    positives = sum(labels.values())
    negatives = len(labels)-positives
    sensitivity = tps/positives
    specificity = tns/negatives
    spositives = sum(sys_out.values())
    snegatives = len(sys_out)-spositives
    PPV = tps/spositives
    NPV = tns/snegatives    
    out_data[ '+'.join([item[i:i+2] for i in range(0,len(item),2)]) ] = [R['AUC'],sensitivity,specificity,PPV,NPV]        
    print("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format( '+'.join([item[i:i+2] for i in range(0,len(item),2)]) ,R['AUC'],sensitivity,specificity,PPV,NPV))

#%%
if True:
    item='Br+Co+Sp+Sy'
    R = pickle.load(open(rootpath+'fusion/BrCoSpSy_results.pkl','rb'))
    inds=np.where(1-R['FPR']>=sthreshold)[0]    
    threshold = R['thresholds'][inds[0]]
    scores = to_dict(rootpath+'fusion/BrCoSpSy_scores.txt')
    labels=to_dict(rootpath+'data/test_labels',{'p':1,'n':0})
    sys_out = {}
    for i in scores: 
        if scores[i]>threshold: 
            sys_out[i]=1
        else: 
            sys_out[i]=0
    tps=0;tns=0
    for i in labels:
        if sys_out[i]==labels[i]:
            if labels[i]==0:
                tns+=1
            if labels[i]==1:
               tps+=1 
    positives = sum(labels.values())
    negatives = len(labels)-positives
    sensitivity = tps/positives
    specificity = tns/negatives
    spositives = sum(sys_out.values())
    snegatives = len(sys_out)-spositives
    PPV = tps/spositives
    NPV = tns/snegatives    
    out_data[item] = [R['AUC'],sensitivity,specificity,PPV,NPV]        
    print("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(item,R['AUC'],sensitivity,specificity,PPV,NPV))
    
#%%
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=[18,5])
ax = plt.subplot(1,1,1)
FS = 12
bwidth = 0.1

V = list(out_data.values())
K = list(out_data.keys())
clr = ['tab:orange','tab:green','tab:red','tab:blue']

ax.bar([i-0.3 for i in range(len(V))],[V[i][1] for i in range(len(V))],width=bwidth,label='Sensitivity')
ax.bar([i-0.1 for i in range(len(V))],[V[i][3] for i in range(len(V))],width=bwidth,label='PPV')
ax.bar([i+0.1 for i in range(len(V))],[V[i][0] for i in range(len(V))],width=bwidth,label='AUC')
ax.bar([i+0.3 for i in range(len(V))],[V[i][4] for i in range(len(V))],width=bwidth,label='NPV')

ax.plot([3.5,3.5],[-0.015,1.15],linestyle='-',color='black',linewidth=1)
ax.plot([7.5,7.5],[-0.015,1.15],linestyle='-',color='black',linewidth=1)
#plt.plot([13.5,13.5],[0,1.2],linestyle='--',color='black')

plt.text(1.5,1.0,'Modalities',ha='center', fontsize=FS+2)
plt.text(5.5,1.0,'Acoustic fusion',ha='center', fontsize=FS+2)
plt.text(11.0,1.0,'Symptom and Acoustic fusion',ha='center', fontsize=FS+2)

ax.set_xticks([i for i in range(len(V))])
labels = ['Breathing \n (Br)','Cough \n (Co)','Speech \n (Sp)', 'Symptoms \n (Sy)', 'Br+Co', 'Br+Sp', 'Co+Sp', 'Br+Co+Sp',
            'Br+Sy', 'Co+Sy','Sp+Sy','Br+Co+Sy','Br+Sp+Sy','Co+Sp+Sy','Br+Co+Sp+Sy']
ax.set_xticklabels(labels,rotation=0,fontsize=FS-2)
ax.set_ylabel('')
plt.yticks(fontsize=FS)
# ax.set_aspect(6)
plt.legend(ncol=4,bbox_to_anchor=[0.5,1.25],loc='upper center', frameon=False,fontsize=FS+2)
plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xlim([-0.5,14.5])
plt.ylim([-0.015,1.1])

if savefig: plt.savefig(plotsdir+'/core_fusion_results.pdf',bbox_inches='tight')