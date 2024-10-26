#!/bin/bash 

# profile vllm server
<<<<<<< HEAD
python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=500 --use_burstgpt --prompt_num=500 --scale=1100 --burstgpt_path=../data/BurstGPT_1.csv --model_path=/root/autodl-tmp/Llama-3.2-1B --max_tokens=64 --detail_log_path=./logs/detail_log_scale1100_prompt500.json --gpu_log_path=./logs/gpu_log_scale1100_prompt500.json
=======
python profile_vllm_server.py --port=8000 --temperature=0 --data_path=preprocess_data/shareGPT.json --stream --surplus_prompts_num=500 --use_burstgpt --prompt_num=500 --scale=1000 --burstgpt_path=../data/BurstGPT_1.csv --model_path=/root/autodl-tmp/Llama-3.2-1B --max_tokens=8 --detail_log_path=./logs/detail_log_scale1000_prompt50.json --gpu_log_path=./logs/gpu_log_scale1000_prompt50.json
>>>>>>> 1aa4dd0c41f3668df3123a097608ec1b41df3067
