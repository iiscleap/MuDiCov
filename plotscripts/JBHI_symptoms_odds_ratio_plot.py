#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:58:42 2021

@author: srikanthr and neeraj
"""
import pandas as pd
import pickle
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import upsetplot
import numpy as np

plt.rcParams.update({'font.size': 12})

root_path='../'
path_store_figure = './plots/'
fig_save = True

## symptoms
symptoms_keys=open(root_path+'data/symptoms')
symptoms_keys = [line.strip() for line in symptoms_keys]
symptoms_keys_map = {'fever':'Fever','cold':'Cold','cough':'Cough',
                     'mp':'Muscle pain','loss_of_smell':'Loss of smell',
                     'st':'Sore throat','ftg':'Fatigue',
                     'diarrhoea':'Diarrhoea'}

#%%
all_data = pd.read_csv(root_path+'data/metadata.csv')
train_list = open(root_path+'data/train_labels').readlines()
train_list = [line.split()[0] for line in train_list]

test_list = open(root_path+'data/test_labels').readlines()
test_list = [line.split()[0] for line in test_list]

all_list = train_list+test_list

all_data = all_data[ all_data.id.isin(all_list)]
all_data.covid_status.replace('positive_asymp','p',inplace=True)
all_data.covid_status.replace('positive_mild','p',inplace=True)
all_data.covid_status.replace('positive_moderate','p',inplace=True)
all_data.covid_status.replace('healthy','n',inplace=True)
all_data.covid_status.replace('no_resp_illness_exposed','n',inplace=True)
all_data.covid_status.replace('resp_illness_not_identified','n',inplace=True)
symptoms_keys = ['cold','fever','cough','st','ftg','mp','loss_of_smell','diarrhoea']
symptoms_labels = ['COLD', 'FEVER', 'COUGH', 'SORE THROAT','FATIGUE', 'MUSCLE PAIN','LOSS OF SMELL', 'DIARRHOEA']
symptoms_dict = {}

i = 0
for key in symptoms_keys:
    all_data[key].fillna(False,inplace=True)
    all_data[key].replace(np.nan,False)
    symptoms_dict[key] = symptoms_labels[i]
    i = i+1


vals_pos = []
vals_neg = []

no_vals_pos = []
no_vals_neg = []

for key in symptoms_keys:
    vals_pos.append(len(all_data[(all_data['covid_status']=='p') & (all_data[key]==True)].values))
    vals_neg.append(len(all_data[(all_data['covid_status']=='n') & (all_data[key]==True)].values))

    no_vals_pos.append(len(all_data[(all_data['covid_status']=='p') & (all_data[key]==False)].values))
    no_vals_neg.append(len(all_data[(all_data['covid_status']=='n') & (all_data[key]==False)].values))

odds_ratio = []
for i in range(len(symptoms_keys)):
    odds_ratio.append((vals_pos[i]/(vals_pos[i]+no_vals_pos[i]))/(vals_neg[i]/(vals_neg[i]+no_vals_neg[i])))

indx_sort = np.argsort(odds_ratio)
odds_ratio = np.array(odds_ratio)[indx_sort]
labels = []
for i in indx_sort:
    labels.append(symptoms_dict[symptoms_keys[i]])

fig = plt.subplots(figsize=[8,6])
ax = plt.subplot(1,1,1)
FS = 12
clr_1 = 'tab:red'
height=.2
ax.barh(np.arange(0,len(labels)),odds_ratio,color=clr_1, alpha=1,height=height)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False) 
ax.set_ylabel('SYMPTOMS',labelpad=10,fontsize=FS)
ax.set_xlabel('ODDS RATIO',fontsize=FS)
for i in range(len(labels)):
    ax.text(odds_ratio[i]+1,i-.1,labels[i],color='black',fontsize=FS-1)
ax.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
plt.xticks(fontsize=FS)
plt.yticks([])
plt.xlim([5,50])
fmt = '.pdf'
if fig_save:
    ax.figure.savefig(path_store_figure+"JBHI_symptoms"+fmt, bbox_inches='tight')
# plt.show()
print('Ploting odds ratio complete!')
