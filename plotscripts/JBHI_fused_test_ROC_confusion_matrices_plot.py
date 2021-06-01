#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:55:23 2021

@author: srikanthr and neeraj
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
#%%
classifier='lr'
arg2='l2'
plotsdir='plots'
rootpath='../'
savefig=True

#%%
folders={}
for category in ['breathing','cough','speech']:
    folders[category]=category+'/results_'+classifier+'/'
folders['symptoms']='symptoms'

#%%
specificity_thresholds=[0.95]
tpr_tnr_data={}
ROCdata={}
for folId in folders:
    R = pickle.load(open(rootpath+folders[folId]+'/test_results.pkl','rb'))

    ROCdata[folId]={'auc':str(round(R['AUC'],2)),'x':R['FPR'],'y':R['TPR'],'label':folId.capitalize()+" ("+folId[:2].capitalize()+")"}    
    sensitivity=[]
    specificity=[]
    for specificity_threshold in specificity_thresholds:
        ind = np.where(1-R['FPR']>specificity_threshold)[0]
        sensitivity.append( R['TPR'][ind[0]])
        specificity.append( 1-R['FPR'][ind[0]])

    tpr_tnr_data[folId]=(folId[:2].upper(),sensitivity,specificity)

R = pickle.load(open(rootpath+'fusion/'+'BrCoSp_results.pkl','rb'))
ROCdata['BR+CO+SP']={'auc':str(round(R['AUC'],2)),'x':R['FPR'],'y':R['TPR'],'label':'Br+Co+Sp'}

sensitivity=[]
specificity=[]
for specificity_threshold in specificity_thresholds:
    ind = np.where(1-R['FPR']>specificity_threshold)[0]
    sensitivity.append( R['TPR'][ind[0]])
    specificity.append( 1-R['FPR'][ind[0]])
tpr_tnr_data['BR+CO+SP']=('BR+CO+SP',sensitivity,specificity)

R = pickle.load(open(rootpath+'fusion/'+'BrCoSpSy_results.pkl','rb'))
ROCdata['BR+CO+SP+SY']={'auc':str(round(R['AUC'],2)),'x':R['FPR'],'y':R['TPR'],'label':'Br+Co+Sp+Sy'}

sensitivity=[]
specificity=[]
for specificity_threshold in specificity_thresholds:
    ind = np.where(1-R['FPR']>specificity_threshold)[0]

    sensitivity.append( R['TPR'][ind[0]])
    specificity.append( 1-R['FPR'][ind[0]])
tpr_tnr_data['BR+CO+SP+SY']=('BR+CO+SP+SY',sensitivity,specificity)

#%% Plotting
fig = plt.subplots(figsize=[7,7])
ax = plt.subplot(1,1,1)
FS = 12
clr = ['tab:orange','tab:green','tab:red','tab:blue','m','navy']

i = 0
for item in ROCdata:
    if i == 5:
        ax.plot(ROCdata[item]['x'],ROCdata[item]['y'],label=
        ROCdata[item]['label']+', AUC='+ROCdata[item]['auc'],linestyle='-',c=clr[i],linewidth=3)
    if i!=5:
        ax.plot(ROCdata[item]['x'],ROCdata[item]['y'],label=ROCdata[item]['label']+', AUC='+ROCdata[item]['auc'],linestyle='-',c=clr[i],linewidth=1)
    i = i+1

ax.plot([0,1],[0,1],linestyle='--', label='chance',c='black',alpha=.5)
leg = ax.legend(loc='lower right', frameon=False, fontsize=FS-2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
plt.xticks(np.arange(0,1.1,0.2),(np.round((np.arange(0,1.1,0.2))*10)/10)[::-1] ,rotation=0,fontsize=FS)
plt.xticks(fontsize=FS);plt.yticks(fontsize=FS)
ax.set_ylabel('SENSITIVITY',fontsize=FS)
ax.set_xlabel('SPECIFICITY',fontsize=FS)
if savefig:
    plt.savefig(plotsdir+'/scorefusion_'+classifier+'_ROCs.pdf',bbox_inches='tight')
#%%

temp = open(rootpath+'data/test_labels')
labels = [line.strip().split(' ')[1] for line in temp]
positives = [item for item in labels if item=='p']
negatives = [item for item in labels if item=='n']
#for idx,sensitivity_threshold in enumerate(sensitivity_thresholds):
for idx,threshold in enumerate(specificity_thresholds):
    outfilenames={'BR':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_BR.pdf',
                  'CO':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_CO.pdf',
                  'SP':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_SP.pdf',
                  'SY':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_SY.pdf',
                  'BR+CO+SP':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_BCSp.pdf',
                  'BR+CO+SP+SY':plotsdir+'/confusion_matrix_'+classifier+'_'+str(threshold)+'_BCSpSy.pdf',}            
    for item in tpr_tnr_data:
        lab,tpr,tnr = tpr_tnr_data[item]
        tpr=tpr[idx];tnr=tnr[idx]
        
        A = np.array([[np.round(tpr*len(positives)),len(negatives)-np.round(tnr*len(negatives))],
                        [len(positives)-np.round(tpr*len(positives)),np.round(tnr*len(negatives))]])

        B = np.array([[tpr,1-tpr],
                        [1-tnr,tnr]])

        fig = plt.subplots(figsize=[7,7])
        ax = plt.subplot(1,1,1)
        FS = 38
        ax.imshow(B,cmap='Blues',vmax=1,vmin=0)
        C = np.round((B>0.7)*B)
        ax.text(0,0,int(A[0,0]),fontsize=FS,ha='center',va='center',c=(C[0,0],C[0,0],C[0,0]))
        ax.text(0,1,int(A[0,1]),fontsize=FS,ha='center',va='center',c=(C[1,0],C[1,0],C[1,0]))
        ax.text(1,0,int(A[1,0]),fontsize=FS,ha='center',va='center',c=(C[0,1],C[0,1],C[0,1]))
        ax.text(1,1,int(A[1,1]),fontsize=FS,ha='center',va='center',c=(C[1,1],C[1,1],C[1,1]))
        
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])

        
        if savefig: plt.savefig(outfilenames[lab],bbox_inches='tight')

#%%