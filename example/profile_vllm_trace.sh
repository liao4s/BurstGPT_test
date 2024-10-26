#!/bin/bash 

prompt=${1:-500}
scale=${2:-1000}
# profile vllm server
python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=$prompt --use_burstgpt --prompt_num=50000 --scale=$scale --burstgpt_path=../data/BurstGPT_1.csv --model_path=/root/autodl-tmp/Llama-3.2-1B --max_tokens=128 --detail_log_path=./logs/detail_log/detail_log_scale${scale}_prompt${prompt}.json --gpu_log_path=./logs/gpu_log/gpu_log_scale${scale}_prompt${prompt}.json --vllm_log_path=./logs/vllm_log/vllm_log_scale${scale}_prompt${prompt}.csv
