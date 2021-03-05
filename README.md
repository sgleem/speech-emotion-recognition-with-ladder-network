# Speech emotion recognition with ladder network
This program was tested on 
- Ubuntu 18.04
- RTX 3090
- CUDA 11.0
- cudnn 8.0.8
- numpy 1.19.5
- pytorch 1.7.0
- opensmile 2.3

All of the implementations are based on the paper, ["Semi-Supervised Speech Emotion Recognition withLadder Networks"](https://arxiv.org/pdf/1905.02921.pdf)

To exectue the whole program, you need to do:
1. Request to use MSP-Podcast database in [here](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html), and download it.
2. If you want to use unlabeled dataset in your training, prepare the dataset that you want to use. (Make sure that all of the files are wav format, and sampled with 16kHz.)
3. Set the directory path in run.sh with respect to the file path in your machine
4. Type 'bash run.sh' in your terminal.

You can test the best model by using this command:
````
python3 -u eval_ladder.py --norm_type=2 --net_type=ladder --task_type=STL --model_path=model/ladder
````

