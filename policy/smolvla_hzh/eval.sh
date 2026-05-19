#!/bin/bash

policy_name=smolvla_hzh
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
remote_host=${6:-localhost}
remote_port=${7:-8000}
smolvla_step=${8:-}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mwebsocket server: ${remote_host}:${remote_port}\033[0m"
if [ -n "${smolvla_step}" ]; then
    echo -e "\033[33msmolvla_step override: ${smolvla_step}\033[0m"
fi

# Run this script from the RoboTwin root directory.

extra_overrides=()
if [ -n "${smolvla_step}" ]; then
    extra_overrides+=(--smolvla_step "${smolvla_step}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --use_websocket True \
    --remote_host ${remote_host} \
    --remote_port ${remote_port} \
    "${extra_overrides[@]}"
