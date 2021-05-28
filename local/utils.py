import numpy as np

def to_dict(filename):
	data = open(filename).readlines()
	D = {}
	for line in data:
		key,val=line.strip().split()
		try:
			val = int(val)
		except:
			try:
				val = float(val)
			except:
				pass
		D[key] = val
	return D

def ArithemeticMeanFusion(scores):

	if len(scores)==1: return scores
	else:
		keys = set(scores[0].keys())
		for i in range(1,len(scores)):
			if keys != set(scores[i].keys()): raise ValueError("Expected all scores to come from same set of participants")

		fused_scores={}
		for key in keys:
			s = [scores[i][key] for i in range(len(scores))]
			sf = sum(s)/len(s)
			fused_scores[key]=sf
	return fused_scores


