#!/bin/bash 

# profile vllm server
python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=500 --use_burstgpt --prompt_num=500 --scale=1000 --burstgpt_path=../data/BurstGPT_1.csv --model_path=/root/autodl-tmp/Llama-3.2-1B --max_tokens=8 --detail_log_path=./logs/detail_log_scale1000_prompt50.json --gpu_log_path=./logs/gpu_log_scale1000_prompt50.json
