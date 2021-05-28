#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:40:19 2020
Modified pn Mon Feb 08 13:21:00 2021

@author: neerajs/srikanthr
"""

import argparse
import numpy as np
import pickle
from sklearn.metrics import auc


def scoring(refs,sys_outs,out_file=None,sensitivities_chosen=[0.5,0.8]):
    """
    inputs::
    refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
    sys_outs: a txt file with a list of scores (probability of being covid positive) for each wav-fileid in the format: <id> <score>
    threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC
    
    outputs::
        
    """    

    thresholds=np.arange(0,1,0.0001)
    # Read the ground truth labels into a dictionary
    data = open(refs).readlines()
    reference_labels={}
    categories = ['n','p']
    for line in data:
        key,val= line.strip().split()
        reference_labels[key]=categories.index(val)

    # Read the system scores into a dictionary
    data = open(sys_outs).readlines()
    sys_scores={}
    for line in data:
        key,val= line.strip().split()
        sys_scores[key]=float(val)
    del data
    
	# Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        print("Expected the score file to have scores for all files in reference and no duplicates/extra entries")
        return None
    #%%

    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((len(reference_labels),len(thresholds)))
    TN = np.zeros((len(reference_labels),len(thresholds)))
    keyCnt=-1
    for key in sys_scores: # Repeat for each recording
        keyCnt+=1
        sys_labels = (sys_scores[key]>=thresholds)*1	# System label for a range of thresholds as binary 0/1
        gt = reference_labels[key]
        
        ind = np.where(sys_labels == gt) # system label matches the ground truth
        if gt==1:	# ground-truth label=1: True positives 
            TP[keyCnt,ind]=1
        else:		# ground-truth label=0: True negatives
            TN[keyCnt,ind]=1
            
    total_positives = sum(reference_labels.values())	# Total number of positive samples
    total_negatives = len(reference_labels)-total_positives # Total number of negative samples
    
    TP = np.sum(TP,axis=0)	# Sum across the recordings
    TN = np.sum(TN,axis=0)
    
    TPR = TP/total_positives	# True positive rate: #true_positives/#total_positives
    TNR = TN/total_negatives	# True negative rate: #true_negatives/#total_negatives
	
    AUC = auc( 1-TNR, TPR )    	# AUC 

    specificities=[]
    sensitivities=[]
    for sens in sensitivities_chosen:
        ind = np.where(TPR>=sens)[0]
        sensitivities.append( TPR[ind[-1]] )
        specificities.append( TNR[ind[-1]] ) 
	# pack the performance metrics in a dictionary to save & return
	# Each performance metric (except AUC) is a array for different threshold values
	# Specificity at 90% sensitivity
    scores={'TPR':TPR,
            'FPR':1-TNR,
            'AUC':AUC,
            'sensitivity':sensitivities,
            'specificity':specificities,
			'thresholds':thresholds}

    if out_file != None:
        with open(out_file,"wb") as f: pickle.dump(scores,f)
    return scores

if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--ref_file','-r',required=True)
	parser.add_argument('--target_file','-t',required=True)
	parser.add_argument('--output_file','-o',default=None)
	args = parser.parse_args()

	scoring(args.ref_file,args.target_file,args.output_file)
