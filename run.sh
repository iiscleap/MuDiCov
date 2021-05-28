stage=0

listsdir=LISTS
datadir=data
featsdir=feats

#set -e
if [ $stage -le 0 ]; then

	if [ ! -d Coswara-Data ];then
		git clone https://github.com/iiscleap/Coswara-Data.git
	fi
	if [ ! -d Coswara-Data/Extracted_data ];then
		cd Coswara-Data
		python extract_data.py
		cd ../
	fi
fi
CoswaraDir=Coswara-Data/Extracted_data
CoswaraMetadata=Coswara-Data/combined_data.csv
if [ $stage -le 1 ]; then
	echo "==== Preparing data folders ====="
	mkdir -p $datadir
	
	cat $listsdir/*_list $listsdir/recovered_ids $listsdir/negatives_after_april2021 | sort | uniq | cut -d' ' -f1 >$datadir/utt_ids
	nfolds=$(ls $listsdir/train_*_list | wc -l)
	echo $nfolds >$datadir/nfolds

	for fold in $(seq 1 $nfolds);do
		mkdir -p $datadir/fold_$fold
		for item in train val;do
			cp $listsdir/${item}_fold_${fold}_list $datadir/fold_$fold/${item}_labels
		done
	done
	cat $datadir/fold_*/train_labels | sort | uniq >$datadir/train_labels
	cat $listsdir/test_list >$datadir/test_labels
		
	for audioname in breathing-deep cough-heavy counting-normal;do
		find $CoswaraDir -name "${audioname}.wav" | awk '{nf=split($1,a,"/");print a[nf-1]" "$1}' > temp
		awk 'FNR==NR {a[$1]; next} ($1 in a)' $datadir/utt_ids temp >$datadir/$audioname.scp
		rm temp  
	done
	mv $datadir/breathing-deep.scp $datadir/breathing.scp
	mv $datadir/cough-heavy.scp $datadir/cough.scp
	mv $datadir/counting-normal.scp $datadir/speech.scp

	awk -F "," 'FNR==NR {a[$1]; next} ($1 in a)' $datadir/utt_ids $CoswaraMetadata >temp
	cat <(head -n 1 $CoswaraMetadata) temp >$datadir/metadata.csv 	

	cp $listsdir/category_to_class $datadir/
	cp $listsdir/symptoms $datadir
	rm temp
fi

if [ $stage -le 2 ];then
	mkdir -p $featsdir
	echo "==== Feature extraction ====="
	for audio in breathing cough speech;do
		echo "for $audio"
		python local/feature_extraction.py $datadir/$audio.scp $featsdir/${audio}.csv
	done
fi

if [ $stage -le 3 ];then
	echo "==== Symptoms ====="
	mkdir -p symptoms
	python local/classifier_on_symptoms.py $datadir symptoms
fi

if [ $stage -le 4 ];then
	echo "==== Audio signals ====="
	mkdir -p breathing
	mkdir -p cough
	mkdir -p speech
	for classifier in lr linearSVM rbfSVM;do 
		python local/classifier_on_audios.py breathing $classifier $datadir $featsdir/breathing.csv breathing &
		python local/classifier_on_audios.py cough $classifier $datadir $featsdir/cough.csv cough &
		python local/classifier_on_audios.py speech $classifier $datadir $featsdir/speech.csv speech 
		wait
	done
fi

if [ $stage -le 5 ];then
	echo "==== Score fusion ====="
	mkdir -p fusion
	python local/score_fusion.py $datadir fusion
fi

echo "Done"
cat fusion/RESULTS
