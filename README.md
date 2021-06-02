
#############################################################################################
#                                                                                           # 
# Multi-modal Point-of-Care Diagnostic Methods for COVID-19 Based on Acoustics and Symptoms #
#                            	                											#
#                                                                                           #  
#############################################################################################

---------
1. About:
---------

This software reproduces the results in the manuscript "Multi-modal point-of-care diagnostic 
methods for COVID-19 based on acoustics and symptoms", submitted to IEEE Journal of Biomedical
and Health Informatics. 
A preprint of the manuscript is available at https://arxiv.org/abs/2106.00639

-----------------------
2. Directory structure:
-----------------------
.
├── LICENSE.md
├── README.md
├── local
│   ├── classifier_on_audios.py
│   ├── classifier_on_symptoms.py
│   ├── feature_extraction.py
│   ├── score_fusion.py
│   ├── scoring.py
│   └── utils.py
├── plotscripts
│   ├── JBHI_dataset_metadata_plot.py
│   ├── JBHI_symptoms_odds_ratio_plot.py
│   ├── JBHI_lr_svm_val_test_ROCs_plot.py
│   ├── JBHI_performance_summary_plot.py
│   ├── JBHI_fused_test_ROC_confusion_matrices_plot.py
│   ├── JBHI_score_analysis_plot.py
├── LISTS
│   ├── train_fold_[1-5]_list
│   ├── val_fold_[1-5]_list
│   ├── test_list
│   ├── recovered_ids
│   ├── negatives_after_april2021
│   ├── category_to_class
│   └── symptoms
├── run.sh
└── REQUIREMENTS.txt

----------------------
3. Directory contents:
----------------------

- run.sh					    	[ Master (shell) script to run the codes ]

- local/feature_extraction.py   	[ Extract ComParE2016 features using opensmile ]	

- local/classifier_on_audios.py   	[ Train LR, Linear-SVM, RBF-SVM classifiers on audio signals ]

- local/classifier_on_symptoms.py   [ Train decision tree classifier on Syptoms feature ]

- local/score_fusion.py   			[ Score fusion of results from three audio categories and symptoms ]

- local/utils.py   					[ Commom util functions ]

- local/scoring.py                  [ Performance: computes false positives, true positives, etc.,
                                	from ground truth labels and the system scores ]

- plotscripts/JBHI_dataset_metadata_plot.py						[Generate Fig. 1 in the manuscript]

- plotscripts/JBHI_symptoms_odds_ratio_plot.py 					[Generate Fig. 4 in the manuscript]

- plotscripts/JBHI_lr_svm_val_test_ROCs_plot.py 				[Generate Fig. 5 in the manuscript] 

- plotscripts/JBHI_performance_summary_plot.py 					[Generate Fig. 7 in the manuscript]

- plotscripts/JBHI_fused_test_ROC_confusion_matrices_plot.py 	[Generate Figs. 8,9 in the manuscript]

- plotscripts/JBHI_score_analysis_plot.py 						[Generate Fig. 10 in the manuscript]

- REQUIREMENTS.txt              	[ Contains a list of dependencies to run the system ]

--------------
4. How to run:
--------------

- Open shell terminal (Linux), navigate to the directory containing the code
- Type the following and hit enter: 
$ ./run.sh

--------------------
4. Results: Otained for the Logistic regression classifier
--------------------
-------------------------------------------------------------------------------------
[ Sensitivity computed at 95% specificity ] 
-----------------------------------------------------------------
Model				|	AUC		|	Sensitivity (specificity) 	|
-----------------------------------------------------------------
Breathing			|	0.777 	|		37.90 (95.30)			|
-----------------------------------------------------------------
Cough				|	0.740 	|		24.10 (95.30)			|	
-----------------------------------------------------------------
Speech				|	0.789 	|		27.60 (95.30)			|
-----------------------------------------------------------------
Symtpoms			|	0.802 	|		55.20 (95.70)			|
-----------------------------------------------------------------
Acoustics			|	0.842 	|		44.80 (95.30)			|
-----------------------------------------------------------------
Acoustics & Symptoms|	0.924 	|		69.00 (95.30)			|
-----------------------------------------------------------------

* depending on the version of python packages in your system, the performance may be little different in
the decimal places

--------------
7. Citation
--------------
If you use this software in your work, please cite the relevant paper. 

@misc{chetupalli2021multimodal,
      title={{Multi-modal Point-of-Care Diagnostics for COVID-19 Based On Acoustics and Symptoms}}, 
      author={Srikanth Raj Chetupalli and Prashant Krishnan and Neeraj Sharma and Ananya Muguli and Rohit Kumar and Viral Nanda and Lancelot Mark Pinto and Prasanta Kumar Ghosh and Sriram Ganapathy},
      year={2021},
      eprint={2106.00639},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

--------------
8. Contact Us:
--------------

Please reach out to Sriram Ganapathy, Assistant professor, IISc for any queries.


--------------
9. Authors:
--------------

- Srikanth Raj Chetupalli | Postdoctoral Researcher, IISc, Bangalore
- Prashant Krishnan | Research Assistant, IISc, Bangalore
- Neeraj Kumar Sharma | CV Raman Postdoctoral Researcher, IISc, Bangalore
- Ananya Muguli | Research Assistant, IISc, Bangalore
- Rohit Kumar | Research Associate, IISc, Bangalore
- Dr Viral Nanda | P. D. Hinduja National Hospital and Medical Research Center, Mumbai
- Dr. Lancelot Mark Pinto | P. D. Hinduja National Hospital and Medical Research Center, Mumbai
- Prasanta Kumar Ghosh | Associate Professor, IISc, Bangalore
- Sriram Ganapathy | Assistant Professor, IISc, Bangalore

#############################################################################################
