#!/bin/bash 

python -m vllm.entrypoints.api_server --model ~/autodl-tmp/Llama-3.2-1B --gpu_memory_utilization=0.4
