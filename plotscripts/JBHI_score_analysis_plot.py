#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:24:32 2021

@author: srikanthr and neeraj
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd

savefig=True
plotsdir='plots'
rootpath='../'
best_val_auc_models={}
for category in ['breathing','cough','speech']:
    best_val_auc_models[category]=category+'/results_lr' 
best_val_auc_models['symptoms']='symptoms'
#%%

all_data = pd.read_csv(rootpath+'/data/metadata.csv')
symptoms_keys = ['fever','cold','cough','mp','loss_of_smell','st','ftg','diarrhoea']

recovered_ids = open(rootpath+'LISTS/recovered_ids').readlines()
recovered_ids = [line.strip() for line in recovered_ids]

negatives_after_april2021 = open(rootpath+'LISTS/negatives_after_april2021').readlines()
negatives_after_april2021 = [line.strip() for line in negatives_after_april2021]

recovered_meta_data = all_data[all_data.id.isin(recovered_ids)]
recovered_meta_data = recovered_meta_data[['id']+symptoms_keys]
recovered_meta_data.reset_index()
for key in symptoms_keys:
    recovered_meta_data[key].fillna(False,inplace=True)

obsset_meta_data = all_data[all_data.id.isin(negatives_after_april2021)]
obsset_meta_data = obsset_meta_data[['id']+symptoms_keys]
obsset_meta_data.reset_index()
for key in symptoms_keys:
    obsset_meta_data[key].fillna(False,inplace=True)

temp = open(rootpath+'data/test_labels').readlines()
positive_ids = [line.split()[0] for line in temp if line.strip().split()[1]=='p']
negative_ids = [line.split()[0] for line in temp if line.strip().split()[1]=='n']


all_scores={}
for audiocategory in ['breathing','cough','speech','symptoms']:
    temp = open(rootpath+best_val_auc_models[audiocategory]+'/test_scores.txt')
    pscores={}
    nscores={}
    for line in temp:
        key,val= line.strip().split()
        if key in positive_ids: 
            pscores[key]=float(val)
        elif key in negative_ids:
            nscores[key]=float(val)

    model = pickle.load(open(rootpath+best_val_auc_models[audiocategory]+'/model.pkl','rb')) 
    scaler = model['scaler']
    classifier = model['classifier']
    if audiocategory=='symptoms':
        rscores={}
        for idx,item in recovered_meta_data.iterrows():
            pid = item['id']
            f = [item[key]*1 for key in symptoms_keys]
            sc=classifier.predict_proba(np.array(f,ndmin=2))
            rscores[pid]=sc[0][1]
        ascores={}
        for idx,item in obsset_meta_data.iterrows():
            pid = item['id']
            f = [item[key]*1 for key in symptoms_keys]
            sc=classifier.predict_proba(np.array(f,ndmin=2))
            ascores[pid]=sc[0][1]
    else:
        temp = pd.read_csv(rootpath+'feats/'+audiocategory+'.csv')
        temp = temp [ temp['file_name'].isin(recovered_ids) ]
        temp = temp.to_numpy()
        features = {}
        for line in temp:
            features[line[-1]]=line[1:-1]
            
        rscores={}
        for rid in recovered_ids:
            score = classifier.predict_proba(scaler.transform(features[rid].reshape(1,-1)))
            rscores[rid]=score[0][1]        

        temp = pd.read_csv(rootpath+'feats/'+audiocategory+'.csv')
        temp = temp [ temp['file_name'].isin(negatives_after_april2021) ]
        temp = temp.to_numpy()
        features = {}
        for line in temp:
            features[line[-1]]=line[1:-1]
            
        ascores={}
        for rid in negatives_after_april2021:
            score = classifier.predict_proba(scaler.transform(features[rid].reshape(1,-1)))
            ascores[rid]=score[0][1]  
        
    all_scores[audiocategory] = (nscores,pscores,rscores,ascores)
    
#%%
fusion_scores=[]
for i in range(4):
    scores={}
    keys=list(all_scores['breathing'][i].keys())
    for key in keys:
        s=[]
        for item in ['breathing','cough','speech']: s.append(all_scores[item][i][key])
        s = sum(s)/len(s)
        scores[key]=s
    fusion_scores.append(scores)
all_scores['BR+CO+SP']= (fusion_scores[0],fusion_scores[1],fusion_scores[2],fusion_scores[3])

fusion_scores=[]
for i in range(4):
    scores={}
    keys=list(all_scores['breathing'][i].keys())
    for key in keys:
        s=[]
        for item in ['breathing','cough','speech','symptoms']: s.append(all_scores[item][i][key])
        s = sum(s)/len(s)
        scores[key]=s
    fusion_scores.append(scores)
all_scores['BR+CO+SP+SY']= (fusion_scores[0],fusion_scores[1],fusion_scores[2],fusion_scores[3])
#%%
plotset=['breathing','cough','speech','BR+CO+SP','BR+CO+SP+SY']
plotset_label=['Breathing (Br)','Cough (Co)','Speech (Sp)','Br+Co+Sp','Br+Co+Sp+Sy']
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=[10,5])
ax = plt.subplot(1,1,1)
FS = 14
bwidth = 0.1

clr = ['tab:green','tab:red','tab:pink','tab:orange']
p1=ax.boxplot([list(all_scores[item][0].values()) for item in plotset], 
               positions=[i-0.3 for i in range(len(plotset))], 
               widths=bwidth, notch=True, patch_artist=True, showfliers=False)
clr_1 = [clr[0]]*len(plotset_label)
for item,color in zip(p1['boxes'],clr_1):
    item.set_facecolor(color)

p2=ax.boxplot([list(all_scores[item][1].values()) for item in plotset], 
               positions=[i-0.1 for i in range(len(plotset))], 
               widths=bwidth, notch=True, patch_artist=True, showfliers=False)
clr_1 = [clr[1]]*len(plotset_label)
for item,color in zip(p2['boxes'],clr_1):
    item.set_facecolor(color)

p3=ax.boxplot([list(all_scores[item][2].values()) for item in plotset], 
               positions=[i+0.1 for i in range(len(plotset))], 
               widths=bwidth, notch=True, patch_artist=True, showfliers=False)
clr_1 = [clr[2]]*len(plotset_label)
for item,color in zip(p3['boxes'],clr_1):
    item.set_facecolor(color)

p4=ax.boxplot([list(all_scores[item][3].values()) for item in plotset], 
               positions=[i+0.3 for i in range(len(plotset))], 
               widths=bwidth, notch=True, patch_artist=True, showfliers=False)
clr_1 = [clr[3]]*len(plotset_label)
for item,color in zip(p4['boxes'],clr_1):
    item.set_facecolor(color)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
plt.plot([0.5,0.5],[0,1],linestyle='-',c='black',alpha=0.4,linewidth=1)
plt.plot([1.5,1.5],[0,1],linestyle='-',c='black',alpha=0.4,linewidth=1)
plt.plot([2.5,2.5],[0,1],linestyle='-',c='black',alpha=0.4,linewidth=1)
plt.plot([3.5,3.5],[0,1],linestyle='-',c='black',alpha=0.4,linewidth=1)
plt.gca().set_xticks([i for i in range(len(plotset))])
plt.gca().set_xticklabels([item for item in plotset_label],fontsize=FS)
plt.yticks(fontsize=FS-2)
plt.ylabel('Probability',fontsize=FS)
plt.ylim([0,1])
plt.xlim([-0.5,len(plotset)-0.5])
if savefig: plt.savefig(plotsdir+'/'+'score_distribution.pdf',bbox_inches='tight')
#%%

fusion_key='BR+CO+SP'

plt.figure()

naascores=all_scores[fusion_key][3]
metadata=pd.read_csv(rootpath+'data/metadata.csv')


test_positive_keys=list(all_scores['breathing'][1].keys())
test_positives= metadata[ metadata.id.isin(test_positive_keys)]
totlen=0
for item in test_positives.covid_status.unique():
    temp = test_positives[test_positives.covid_status==item]
    temp=temp.id.to_list()
    sc=[]
    for key in temp:
        sc.append(all_scores[fusion_key][1][key])
    totlen+=len(sc)    

test_negative_keys=list(all_scores['breathing'][0].keys())
test_negatives= metadata[ metadata.id.isin(test_negative_keys)]
totlen=0
A={}
for item in test_negatives.covid_status.unique():
    temp = test_negatives[test_negatives.covid_status==item]
    temp=temp.id.to_list()
    sc=[]
    for key in temp:
        sc.append(all_scores[fusion_key][0][key])
    A[item]=sc



test_negative_keys=list(all_scores['breathing'][3].keys())
test_negatives= metadata[ metadata.id.isin(test_negative_keys)]
totlen=0
B={}
for item in test_negatives.covid_status.unique():
    temp = test_negatives[test_negatives.covid_status==item]
    temp=temp.id.to_list()
    sc=[]
    for key in temp:
        sc.append(all_scores[fusion_key][3][key])
    B[item]=sc

plt.figure()
fig = plt.subplots(figsize=[7,5])
ax = plt.subplot(1,1,1)
FS = 12

p1=plt.boxplot([list(all_scores[fusion_key][0].values()),
                list(all_scores[fusion_key][1].values()),
                list(naascores.values())],
               positions=[0.8,1,1.2], widths=bwidth,notch=True,patch_artist=True, showfliers=False)
colors=['tab:green','tab:red','tab:orange']
for item,color in zip(p1['boxes'],colors):
    item.set_facecolor(color)
p2=plt.boxplot([B[item] for item in B],positions=[1.8,2,2.2],widths=bwidth,notch=True,patch_artist=True, showfliers=False)
colors=['tab:orange','tab:orange','tab:orange']
alphas=[0.8,0.6,0.4]
for item,color,alpha in zip(p2['boxes'],colors,alphas):
    item.set_facecolor(color)
    item.set_alpha(alpha)

p3=plt.boxplot([A[item] for item in A],positions=[-0.2,0,0.2],widths=bwidth,notch=True,patch_artist=True, showfliers=False)
colors=['tab:green','tab:green','tab:green']
alphas=[0.8,0.6,0.4]
for item,color,alpha in zip(p3['boxes'],colors,alphas):
    item.set_facecolor(color)
    item.set_alpha(alpha)
    
plt.gca().set_xticklabels(['non-COVID','COVID','Obs. Set\n (non-COVID)','Healthy','Exposed','Resp. Ail','Healthy','Exposed','Resp. Ail'],rotation=90,fontsize=FS)
plt.ylim([0.0,1.0])
plt.xlim([-0.5,2.5])
plt.yticks(fontsize=FS)
plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
plt.plot([0.5,0.5],[0,1],linestyle='--',c='black',alpha=0.5)
plt.plot([1.5,1.5],[0,1],linestyle='--',c='black',alpha=0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


if savefig: plt.savefig(plotsdir+'/'+'score_distribution_negatives_before_after_Apr2021.pdf',bbox_inches='tight')
