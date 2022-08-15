#!/bin/bash

OpenSmileConfigPath=$1 #/home/kyunster/Lib/opensmile/config
corpus_dir=$2 #/media/kyunster/hdd/corpus/MSP_Podcast_1.7/unlabeled_1.8_clean
hld_path=$3
find ${corpus_dir} -name "*.wav" > filelist.txt
filelist=( $( cat filelist.txt ))

mkdir -p $hld_path

for wav_path in ${filelist[@]};
do
    fname=`echo $wav_path | rev | awk '{split($0,a,"/"); print a[1]}' | rev`
    oname=`echo $fname | awk '{split($0,a,"."); print a[1]}'`.txt
    echo $fname

    if [ ! -f ${hld_path}/${oname} ]; then 
        SMILExtract -C $OpenSmileConfigPath/IS13_ComParE.conf -I $wav_path -O ${hld_path}/${oname} -l 0 || exit 1;
    fi
done

rm filelist.txt