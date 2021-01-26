#!/bin/bash

#########################################################
# Set the directory path
#########################################################
corpus_dir=/media/kyunster/hdd/corpus/MSP_Podcast_1.8
opensmile_dir=/home/kyunster/Lib/opensmile
##########################################################
feature_dir=./feature/labeled
data_dir=./data/labeled
log_dir=./log
model_dir=./model/ladder

################ Preprocess ################
bash extract_feat.sh ${opensmile_dir}/config ${corpus_dir}/Audios ${feature_dir}
python3 -u preprocess_hld.py $feature_dir data_pre/labeled
python3 -u prepare_data.py data_pre/labeled ${data_dir} ${corpus_dir}/Labels/labels_concensus.json

rm -rf data_pre

# Run this scirpt if you have unlabeled dataset for training.
## unlabel_dir=/media/kyunster/hdd/corpus/MSP_Podcast_1.8/unlabeled # Make sure that all unlabeled wav files are in this folder and sampled with 16kHz.
## unlabel_feature_dir=./feature/unlabeled
## unlabel_data_dir=./data/unlabeled

## bash extract_unlabel.sh ${opensmile_dir}/config ${unlabel_dir}
## python3 -u preprocess_hld.py ${unlabel_feature_dir} data_pre/unlabeled
## python3 -u prepare_unlabel.py data_pre/unlabeled ${unlabel_data_dir}
## rm -rf data_pre

################ Training ################
mkdir -p $log_dir
python3 -u train_ladder.py --norm_type=2 --net_type=ladder --task_type=STL \
    --model_path=${model_path} --seed=0 > ${log_dir}/train_log.txt || exit 1;
# Training with unlabel dataset
# python3 -u train_ladder.py --norm_type=2 --net_type=ladder --task_type=STL \
#     --model_path=${model_path} --seed=0 --use-unlabel > ${log_dir}/train_log.txt || exit 1;
    
################ Evaluation ################
python3 -u eval_ladder.py --norm_type=2 --net_type=ladder --task_type=STL \
    --model_path=${model_path} || exit 1;