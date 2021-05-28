#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:58:56 2021

@author: srikanthr
"""

from utils import *
import numpy as np
import sys
from scoring import *
#%%

datadir=sys.argv[1]
outdir=sys.argv[2]

indirs={'breathing':'breathing/results_lr','cough':'cough/results_lr','speech':'speech/results_lr','symptoms':'symptoms'}

outdata=[]
outdataformat='{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n'
#%%
scores={}
for item in indirs:
    scores[item]=(to_dict(indirs[item]+'/test_scores.txt'))

    R=pickle.load(open(indirs[item]+'/test_results.pkl','rb'))
    outdata.append(outdataformat.format(item,R['AUC'],R['sensitivity'][0],R['specificity'][0],R['sensitivity'][1],R['specificity'][1]))
#%%
fileId='BrCoSpSy'
fused_scores=ArithemeticMeanFusion([scores[item] for item in ['breathing','cough','speech','symptoms'] ])
with open(outdir+'/'+fileId+'_scores.txt','w') as f:
    for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))
R=scoring(datadir+'/test_labels',outdir+'/'+fileId+'_scores.txt',outdir+'/'+fileId+'_results.pkl')
outdata.append(outdataformat.format(fileId,R['AUC'],R['sensitivity'][0],R['specificity'][0],R['sensitivity'][1],R['specificity'][1]))
#%%
pairs_ids = [('breathing','cough','BrCo'), ('breathing','speech','BrSp'),
             ('breathing','symptoms','BrSy'),('cough','speech','CoSp'),
             ('symptoms','cough','CoSy'),('speech','symptoms','SpSy'),]

for item in pairs_ids:
    i1,i2,fileId=item
    fused_scores=ArithemeticMeanFusion([scores[i1],scores[i2]])
    
    with open(outdir+'/'+fileId+'_scores.txt','w') as f:
        for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))
    R=scoring(datadir+'/test_labels',outdir+'/'+fileId+'_scores.txt',outdir+'/'+fileId+'_results.pkl')
    outdata.append(outdataformat.format(fileId,R['AUC'],R['sensitivity'][0],R['specificity'][0],R['sensitivity'][1],R['specificity'][1]))
#%%
triplets_ids = [('breathing','cough','speech','BrCoSp'),
                ('breathing','cough','symptoms','BrCoSy'),
                ('breathing','speech','symptoms','BrSpSy'),
                ('cough','speech','symptoms','CoSpSy')]

for item in triplets_ids:
    i1,i2,i3,fileId=item
    fused_scores=ArithemeticMeanFusion([scores[i1],scores[i2],scores[i3]])
    
    with open(outdir+'/'+fileId+'_scores.txt','w') as f:
        for item in fused_scores: f.write('{} {}\n'.format(item,fused_scores[item]))
    R=scoring(datadir+'/test_labels',outdir+'/'+fileId+'_scores.txt',outdir+'/'+fileId+'_results.pkl')
    outdata.append(outdataformat.format(fileId,R['AUC'],R['sensitivity'][0],R['specificity'][0],R['sensitivity'][1],R['specificity'][1]))

#%%
with open(outdir+'/RESULTS','w') as f:
    f.write('id\tAUC\tSp.50\tSe.50\tSp.80\tSe.80\n')
    for item in outdata: f.write(item)
