import opensmile
import librosa
import pandas as pd
import numpy as np
import os,sys
from tqdm import tqdm

config={'sampling_rate':44100, 'threshold':0.0001, 'start_end_sil_length':100, 'silence_margin':50 }

def compute_SAD(sig,config):
	# Speech activity detection based on sample thresholding
	# Expects a normalized waveform as input
	# Uses a margin of at the boundary
    fs = int(config['sampling_rate'])
    sad_thres = float(config['threshold'])
    sad_start_end_sil_length = int(int(config['start_end_sil_length'])*1e-3*fs)
    sad_margin_length = int(int(config['silence_margin'])*1e-3*fs)

    sample_activity = np.zeros(sig.shape)
    sample_activity[np.power(sig,2)>sad_thres] = 1
    sad = np.zeros(sig.shape)
    for i in range(len(sample_activity)):
        if sample_activity[i] == 1:
            sad[i-sad_margin_length:i+sad_margin_length] = 1
    sad[0:sad_start_end_sil_length] = 0
    sad[-sad_start_end_sil_length:] = 0
    return sad

#%%
wav_scp=sys.argv[1]
out_name=sys.argv[2]

file_list = open(wav_scp).readlines()
file_list = [f.split() for f in file_list]
    
data = pd.DataFrame([])
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)
    
for fil in tqdm(file_list):
    file_name, path = fil
    signal, sampling_rate = librosa.load(path, sr=44100)
    signal = signal/max(abs(signal))
    sad = compute_SAD(signal,config)
    ind = np.where(sad>0)[0]
    signal = signal[min(ind):max(ind)]
    d_feats = smile.process_signal(signal,sampling_rate)
    d_feats['file_name'] = file_name
    data = data.append(d_feats, ignore_index=True)
    del d_feats        
data.to_csv(out_name)    
