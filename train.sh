#!/bin/bash

log_dir=log/0125
mkdir -p ${log_dir}

# for seed in `seq 0 9`; do
#     for task_type in MTL; do
#         python3 -u train.py --data_type=clean --feature_type=HLD \
#                 --norm_type=2 --net_type=MLP --task_type=${task_type} \
#                 --model_path=model/clean+Baseline/MLP_${task_type}_${seed} --seed=${seed} > ${log_dir}/MLP_${task_type}_label_${seed}.txt || exit 1;
#     done
# done

# for seed in `seq 0 9`; do
#     for task_type in STL; do
#         python3 -u train_ladder.py --data_type=clean --feature_type=HLD \
#                 --norm_type=2 --net_type=ladder_old --task_type=${task_type} \
#                 --model_path=model/exp2/label/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/ladder_${task_type}_label_${seed}.txt || exit 1;
    
#         for unlab_type in clean 10db 5db 0db; do
#             python3 -u train_ladder.py --data_type=clean --feature_type=HLD \
#                 --norm_type=2 --net_type=ladder_old --task_type=${task_type} --unlabel_type=${unlab_type} \
#                 --model_path=model/exp2/${unlab_type}/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/ladder_${task_type}_${unlab_type}_${seed}.txt || exit 1;
#         done
#     done
# done

# for seed in `seq 0 9`; do
#     for task_type in STL; do
#         python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
#              --norm_type=2 --net_type=ladder_grl --task_type=${task_type} \
#              --model_path=model/Ladder+GRL/label/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/grl_${task_type}_label_${seed}.txt || exit 1;
#         for unlab_type in clean 10db 5db 0db; do
#             python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
#                 --norm_type=2 --net_type=ladder_grl --task_type=${task_type} --unlabel_type=${unlab_type} \
#                 --model_path=model/Ladder+GRL/${unlab_type}/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/grl_${task_type}_${unlab_type}_${seed}.txt || exit 1;
#         done
#     done
# done

# for seed in `seq 0 9`; do
#     for task_type in STL; do
#         python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
#              --norm_type=2 --net_type=ladder_orth --task_type=${task_type} \
#              --model_path=model/Ladder+orth/label/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/orth_${task_type}_label_${seed}.txt || exit 1;
#         for unlab_type in clean 10db 5db 0db; do
#             python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
#                 --norm_type=2 --net_type=ladder_orth --task_type=${task_type} --unlabel_type=${unlab_type} \
#                 --model_path=model/Ladder+orth/${unlab_type}/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/orth_${task_type}_${unlab_type}_${seed}.txt || exit 1;
#         done
#     done
# done


for seed in `seq 0 9`; do
    for task_type in STL; do
         for unlab_type in clean 10db 5db 0db; do
            python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
                --norm_type=2 --net_type=ladder_orth --task_type=${task_type} --unlabel_type=${unlab_type} \
                --model_path=model/Ladder+orth/${unlab_type}/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/orth_${task_type}_${unlab_type}_${seed}.txt || exit 1;
        done
    done
done

for seed in 9; do
    for task_type in STL; do
        python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
             --norm_type=2 --net_type=ladder_separate --task_type=${task_type} \
             --model_path=model/Ladder+separate/label/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/separate_${task_type}_label_${seed}.txt || exit 1;
    done
done

for seed in `seq 0 9`; do
    for task_type in STL; do
        python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
             --norm_type=2 --net_type=ladder_separate --task_type=${task_type} \
             --model_path=model/Ladder+separate/label/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/separate_${task_type}_label_${seed}.txt || exit 1;
        for unlab_type in clean 10db 5db 0db; do
            python3 -u train_decoupled_ladder.py --data_type=clean --feature_type=HLD \
                --norm_type=2 --net_type=ladder_separate --task_type=${task_type} --unlabel_type=${unlab_type} \
                --model_path=model/Ladder+separate/${unlab_type}/ladder_${task_type}_${seed} --seed=${seed} > ${log_dir}/separate_${task_type}_${unlab_type}_${seed}.txt || exit 1;
        done
    done
done

