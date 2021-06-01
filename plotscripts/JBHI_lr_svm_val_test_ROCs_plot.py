#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:24:32 2021

@author: srikanthr and neeraj
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import matplotlib.font_manager

savefig = True
plotsdir = 'plots'
rootpath='../'
#%%
for audiocategory in ['breathing','cough','speech']:

    figure_prefix = plotsdir+'/'+audiocategory+'_Functionals_test_ROCCurves'

#%% Read Data
    best_val_auc_models = {}
    AUCsOfClassfiers = {}
    
    cAUCs={}
    bestAUC=0
    for c in range(-7,8):
        resultsdir = rootpath+audiocategory+'/results_lr_c'+str(c)
        
        fold_aucs = []
        for fold in range(1,6):
            R = pickle.load(open(resultsdir+'/fold_'+str(fold)+'/val_results.pkl','rb'))
            fold_aucs.append(R['AUC'])
        auc = sum(fold_aucs)/len(fold_aucs)
        cAUCs[c] = auc
        if auc>bestAUC: 
            best_val_auc_models['lr'] = resultsdir
            bestAUC = auc
    AUCsOfClassfiers['lr']=cAUCs
    
    cAUCs={}
    bestAUC=0
    for c in range(-7,8):
        resultsdir = rootpath+audiocategory+'/results_linearSVM_c'+str(c)
        
        fold_aucs = []
        for fold in range(1,6):
            R = pickle.load(open(resultsdir+'/fold_'+str(fold)+'/val_results.pkl','rb'))
            fold_aucs.append(R['AUC'])
        auc = sum(fold_aucs)/len(fold_aucs)
        cAUCs[c] = auc
        if auc>bestAUC: 
            best_val_auc_models['lsvm'] = resultsdir
            bestAUC = auc
    AUCsOfClassfiers['lsvm']=cAUCs

    cAUCs={}
    bestAUC=0
    for c in range(-7,8):
        resultsdir = rootpath+audiocategory+'/results_rbfSVM_c'+str(c)        
        fold_aucs = []
        for fold in range(1,6):
            R = pickle.load(open(resultsdir+'/fold_'+str(fold)+'/val_results.pkl','rb'))
            fold_aucs.append(R['AUC'])
        auc = sum(fold_aucs)/len(fold_aucs)
        cAUCs[c] = auc
        if auc>bestAUC: 
            best_val_auc_models['rsvm'] = resultsdir
            bestAUC = auc
    AUCsOfClassfiers['rsvm']=cAUCs
        
#%% Plot ROC curves

    classifiers = {'lr':'LR','lsvm':'Lin-SVM','rsvm':'RBF-SVM'}
    colors = {'lr':'blue','lsvm':"green",'rsvm':'red'}
    dev_rocs_data={};test_rocs_data={}
    ci_rocs_data={}
    for item in best_val_auc_models:
        data_x=[];data_y=[];data_auc=[]
        for fold in range(1,6):        
            R = pickle.load(open(best_val_auc_models[item]+'/fold_'+str(fold)+'/val_results.pkl','rb'))
            data_x.append(R['FPR'].tolist())
            data_y.append(R['TPR'].tolist())
            data_auc.append(R['AUC'])    
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        
        ci_offset = 0.9*np.std(np.array(data_auc))/(len(data_auc)**0.5)
        mauc=np.mean(np.array(data_auc))
        
        dev_rocs_data[item] = {'x':np.mean(data_x,axis=0),'y':np.mean(data_y,axis=0),'label':'{}, AUC {:.2f} [CI: {:.2f}-{:.2f}]'.format(classifiers[item],mauc,mauc-ci_offset,mauc+ci_offset) }
        # dev_rocs_data[item] = {'x':np.mean(data_x,axis=0),'y':np.mean(data_y,axis=0),'label':'{}, AUC {:.2f}'.format(classifiers[item],mauc,mauc-ci_offset,mauc+ci_offset) }
        
        fpr_ci_offset = 0.9*np.std(data_x,axis=0)/np.sqrt(5)
        tpr_ci_offset = 0.9*np.std(data_y,axis=0)/np.sqrt(5)        
        
        ci_rocs_data[item] = {'x1':np.mean(data_x,axis=0)-fpr_ci_offset,'x2':np.mean(data_x,axis=0)+fpr_ci_offset,
                              'y1':np.mean(data_y,axis=0)-tpr_ci_offset,'y2':np.mean(data_y,axis=0)+tpr_ci_offset}
        

        testmodelfolder=best_val_auc_models[item]
        testmodelfolder="_".join(testmodelfolder.split("_")[:-1])        
        R = pickle.load(open(testmodelfolder+'/test_results.pkl','rb'))
        test_rocs_data[item] = {'x':R['FPR'],'y':R['TPR'],'label':'{} {:.2f}'.format(classifiers[item],R['AUC'])}
        
    #%% Plotting   
    fig = plt.subplots(figsize=[4,4])    
    ax = plt.subplot(1,1,1)
    FS = 10
    for item in best_val_auc_models:
        ax.plot(dev_rocs_data[item]['x'],dev_rocs_data[item]['y'], label=dev_rocs_data[item]['label'], c=colors[item],alpha=1,linewidth=1)
        ax.fill_between(dev_rocs_data[item]['x'], ci_rocs_data[item]['y1'],ci_rocs_data[item]['y2'],
                color=colors[item],alpha=0.1)

    # for item in best_val_auc_models:
    #     ax.plot(ci_rocs_data[item]['x1'],ci_rocs_data[item]['y1'], c=colors[item],alpha=.5,linewidth=1)
    #     ax.plot(ci_rocs_data[item]['x2'],ci_rocs_data[item]['y2'], c=colors[item],alpha=.5,linewidth=1)
        
    ax.plot([0,1],[0,1],linestyle='--', label='chance',c='black',alpha=.5)
    ax.legend(loc='lower right', frameon=False,fontsize=FS-2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.2)
    plt.xticks(np.arange(0,1.1,0.2),(np.round((np.arange(0,1.1,0.2))*10)/10)[::-1] ,rotation=0,fontsize=FS)
    plt.xticks(fontsize=FS);plt.yticks(fontsize=FS)
    ax.set_ylabel('SENSITIVITY',fontsize=FS)
    ax.set_xlabel('SPECIFICITY',fontsize=FS)
    if savefig: plt.savefig(figure_prefix+'_dev.pdf',bbox_inches='tight')
    plt.show()
    plt.close()
        
    fig = plt.subplots(figsize=[4,4])    
    ax = plt.subplot(1,1,1)
    for item in best_val_auc_models:
        ax.plot(test_rocs_data[item]['x'],test_rocs_data[item]['y'], label=test_rocs_data[item]['label'], c=colors[item],alpha=1,linewidth=1)
    ax.plot([0,1],[0,1],linestyle='--', label='chance',c='black',alpha=.5)
    ax.legend(loc='lower right', frameon=False,fontsize=FS-2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1,alpha=.2)
    plt.xticks(np.arange(0,1.1,0.2),(np.round((np.arange(0,1.1,0.2))*10)/10)[::-1] ,rotation=0,fontsize=FS)
    plt.xticks(fontsize=FS);plt.yticks(fontsize=FS)
    ax.set_ylabel('SENSITIVITY',fontsize=FS)
    ax.set_xlabel('SPECIFICITY',fontsize=FS)
    if savefig: plt.savefig(figure_prefix+'_test.pdf',bbox_inches='tight')   
    plt.show()     
