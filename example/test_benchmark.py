import aiohttp
import argparse
import asyncio
import json
import numpy as np
import os
import random
import re
import requests
import subprocess
import sys
import resource
import time
from datetime import datetime
import pandas as pd

from profile_server import ServerOnline, Config
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional, Dict
from transformers import AutoTokenizer
from async_request_sender import Context, AysncRequestSender, Metrics, calculate_metrics
import util
'''
example:
    python3 /workspace/dynamo-eval/BurstGPT_test/example/test_benchmark.py \
	--endpoint http://localhost:8000/v1 \
	--dataset /workspace/dynamo-eval/BurstGPT_test/data/BurstGPT_1.csv \
	--tokenizer /models/hub/models--neuralmagic--DeepSeek-R1-Distill-Llama-70B-FP8-dynamic/snapshots/fb3637a1165cec3832958bd72ebbe04021601489 \
	--model fb3637a1165cec3832958bd72ebbe04021601489 \
	--sampling-policy normal \
	--parallel 80 \
	--num-requests 1024 \
    --max-sample-request-tokens 5000
'''

SYS_PROMPT="""
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
TMP_TEST_SHARED_PROMPT = False
PRINT_SAMPLES = 10

def build_from_burstgpt_with_sleep(burstgpt_query_generator, max_sample_request_tokens: int = 10000):
    samples = []
    prompt_num = burstgpt_query_generator.config.prompt_config['prompt_num']
    while len(samples) < prompt_num:
        _prompt, in_num, out_num, sampled_in_num, sampled_out_num, _, _sleep_time = burstgpt_query_generator.queries.get_query()
        if _prompt is None or len(_prompt) == 0 or int(out_num) == 0 or len(_prompt) + int(out_num) > max_sample_request_tokens:
            continue
        if len(samples) % 100 == 0:
            print(f"Processing {len(samples)}th request")
        samples.append((str(_prompt), len(_prompt), int(out_num), int(_sleep_time)))
    average_input_len = sum(sample[1] for sample in samples) / len(samples)
    average_output_len = sum(sample[2] for sample in samples) / len(samples)
    print(f"Requests average input_len={average_input_len}, output_len={average_output_len}")
    return samples

def make_placeholder_prompt(n_tokens: int, unit="hello world", tokenizer = None) -> str:
    """
    通过重复 unit（一个短语或字符），拼出一个长度 >= n_tokens 的文本，
    然后截断到最近的 token 边界，近似达到 n_tokens。
    """
    # 估算：先重复 unit 足够多
    repeat_count = (n_tokens // max(1, len(tokenizer.encode(unit)))) + 5
    text = unit * repeat_count

    # 编码并截断
    tokens = tokenizer.encode(text)
    truncated = tokens[:n_tokens]
    return tokenizer.decode(truncated)

def build_from_burstgpt(csv_path: str = "/root/aking/dynamo-eval/AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv", 
    num_requests: int = 1024, tokenizer = None, max_sample_request_tokens: int = 16384):
    samples = []
    workload_ds = pd.read_csv(csv_path, sep=",", header=0, encoding="utf-8")
    assert num_requests <= len(workload_ds), f"num_requests {num_requests} is larger than the dataset size {len(workload_ds)}"
    for i in range(len(workload_ds)):
        seq_len = int(workload_ds["ContextTokens"][i])
        if seq_len > max_sample_request_tokens:
            continue
        if len(samples) > num_requests:
            break
        item = (make_placeholder_prompt(seq_len, tokenizer=tokenizer), int(workload_ds["ContextTokens"][i]), int(workload_ds["GeneratedTokens"][i]))
        samples.append(item)
    average_input_len = sum(sample[1] for sample in samples) / len(samples)
    average_output_len = sum(sample[2] for sample in samples) / len(samples)
    print(f"Requests average input_len={average_input_len}, output_len={average_output_len}")
    return samples

def parse_online_log_file(filename, num_requests, shuffle = False, max_sample_request_tokens: int = 16384):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()

    request_blocks = re.split(r"=== Line \d+ ===", data)
    if shuffle: 
        random.shuffle(request_blocks)

    samples = []
    for i, request_data in enumerate(request_blocks):
        if "Request Body:" in request_data:
            try:
                request_body_str = request_data.split("Request Body:")[1].split("Output Tokens:")[0].strip()
                request_body = json.loads(request_body_str)
                input_prompt = ""
                output_content = request_data.split("Request Body:")[1].split("Output Tokens:")[1].strip()
                if "prompt" in request_body.keys():
                    input_prompt = request_body["prompt"]
                else:
                    input_prompt = ''.join(str(s["content"]) for s in request_body["messages"])
                if len(input_prompt) + len(output_content) > max_sample_request_tokens: continue
                item = (input_prompt, len(input_prompt), len(output_content))
                samples.append(item)
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Error: {e}")
        if len(samples) >= num_requests: break
    average_input_len = sum(sample[1] for sample in samples) / len(samples)
    average_output_len = sum(sample[2] for sample in samples) / len(samples)
    print(f"Requests average input_len={average_input_len}, output_len={average_output_len}")
    return samples

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")
    if "BurstGPT" in args.dataset:
        server_config = dict()
        prompt_config = dict()
        server_config['stream'] = not args.disable_stream
        server_config['ignore_eos'] = False if args.disable_ignore_eos else True
        server_config['qps'] = 1.0
        server_config['host'] = "localhost"
        server_config['port'] = "8000"
        server_config['temperature'] = 0.8
        server_config['max_tokens'] = None

        prompt_config['seed'] = 1
        prompt_config['surplus_prompts_num'] = args.num_requests
        prompt_config['use_burstgpt'] = True
        prompt_config['burstgpt_path'] = args.dataset
        prompt_config['conv_or_api'] = "conv"
        prompt_config['scale'] = args.burstgpt_scale
        prompt_config['prompt_num'] = args.num_requests
        config = Config(server_config=server_config, prompt_config=prompt_config, served_model_name=args.model)
        print(config)
        burstgpt_query_generator = ServerOnline(model_path=args.tokenizer,
                            data_path="/workspace/dynamo-eval/BurstGPT_test/example/preprocess_data/shareGPT.json",
                            backend="vllm",
                            config=config,
                            )
    random.seed(1)
    np.random.seed(1)
    if not args.model:
        server_model = util.get_model(args.endpoint + "/models")
        if server_model is None and not args.model:
            raise RuntimeError("Failed to query model name from server")
        if not args.model:
            args.model = server_model
        assert args.model == server_model, f"Mismatched model name: {args.model}, {server_model}"
    logger.info(f"Model name: {args.model}")
    # get samples from dataset
    if args.sampling_policy == "nature":
        min_in_len = [args.min_prompt_len] * args.num_requests
        max_in_len = [args.max_prompt_len] * args.num_requests
        min_out_len = [args.min_output_len] * args.num_requests
        max_out_len = [args.max_output_len] * args.num_requests
    elif args.sampling_policy == "fixed":
        min_in_len = [args.fixed_prompt_len] * args.num_requests
        max_in_len = min_in_len
        min_out_len = [args.fixed_output_len] * args.num_requests
        max_out_len = min_out_len
    elif args.sampling_policy == "normal":
        min_in_len = np.rint(np.random.normal(args.prompt_len_mean, args.prompt_len_std, size=args.num_requests)).astype(np.int32)
        max_in_len = min_in_len
        min_out_len = np.rint(np.random.normal(args.output_len_mean, args.output_len_std, size=args.num_requests)).astype(np.int32)
        max_out_len = min_out_len
        
    if args.sampling_policy != "undefined" and args.sampling_policy != "order" and min_in_len is None:
        raise RuntimeError("Invalid input length and output length")
    if args.tokenizer is None:
        tokenizer = None
        args.add_stream_usage = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # load dataset
    if "llama-70b-completions_online_dataset" in args.dataset:
        samples = parse_online_log_file(args.dataset, args.num_requests, max_sample_request_tokens=int(args.max_sample_request_tokens))
    elif "BurstGPT" in args.dataset and "csv" in args.dataset:
        samples = build_from_burstgpt_with_sleep(burstgpt_query_generator)
        args.parallel = args.num_requests
    elif "csv" in args.dataset:
        samples = build_from_burstgpt(args.dataset, args.num_requests, tokenizer)
    elif args.dataset in ["mmlupro", "mbpp"] and args.sampling_policy == "undefined":
        if args.dataset == "mmlupro":
            items = util.load_mmlu_pro_dataset(args.num_requests)
        elif args.dataset == "mbpp":
            items = util.load_mbpp_dataset(args.num_requests)
        else:
            raise RuntimeError(f"Invalid dataset: {args.dataset}")
        samples = []
        for item in items:
            samples.append((item[0], 0, 0))
    else:
        filename, file_extension = os.path.splitext(args.dataset)
        if os.path.isabs(args.dataset) and os.path.exists(args.dataset) and file_extension.lower() == ".json":
            samples = util.load_requests_from_json(tokenizer, args.dataset, args.num_requests, min_in_len, max_in_len, min_out_len, max_out_len)
        else:
            dataset = util.load_hf_dataset(args.dataset)
            dataset_filtered = [] # list of (input, output)
            if hasattr(dataset, "features") and "input" in dataset.features and "instruction" in dataset.features:
                for data in dataset:
                    if data["instruction"] is not None and data["input"] is not None and len(data["instruction"]) + len(data["input"]) > 20:
                        dataset_filtered.append((data["instruction"] + "\n" + data["input"], data["output"] if data["output"] is not None else ""))
            else:
                ## need further checking...
                dataset_filtered = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset if data["conversations"][0]["value"] > 10 and data["conversations"][1]["value"] > 10]
            samples = util.filter_samples_from_dataset(dataset_filtered, tokenizer, args.num_requests, min_in_len, max_in_len, min_out_len, max_out_len)

    logger.info(f"Got {len(samples)} requests")
    if len(samples) == 0:
        raise RuntimeError(f"Failed to load samples from dataset: {args.dataset}")
    while len(samples) < args.num_requests:
        samples.append(samples[random.randint(0, len(samples)-1)])
    contexts = []
    for i in range(len(samples)):
        d = samples[i]
        if i < PRINT_SAMPLES and args.verbose:
            logger.info(f"Request[{i}]: {d[1]} / {d[2]}, {d[0][0: 100]}")
        send_timestamp = 0
        if len(samples[i]) > 3:
            send_timestamp = samples[i][3]
        contexts.append(Context(index=i, prompt=d[0], prompt_len=d[1], max_tokens=d[2], send_timestamp=send_timestamp))

    # send requests async
    extra = {}
    ignore_eos = False if args.disable_ignore_eos else True
    sender = AysncRequestSender(args.endpoint, args.model, args.api_key, SYS_PROMPT if args.add_system_prompt else None, False if args.disable_stream else True, args.add_stream_usage, ignore_eos, args.verbose)
    logger.info("Warmup")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts[0:2], args.api_kind == "chat", 2, extra))
    end_time = time.perf_counter()
    logger.info(f"Warmup fininshed in {end_time - start_time} seconds")
    for i in range(min(2, len(contexts))):
        contexts[i].clean()

    logger.info("Benchmark")
    start_time = time.perf_counter()
    asyncio.run(sender.post_batch_requests_async(contexts, args.api_kind == "chat", args.parallel, extra))
    e2e_duration = time.perf_counter() - start_time
    logger.info(f"Benchmark fininshed in {e2e_duration} seconds")
    # metrics
    metrics, metrics_good = calculate_metrics(tokenizer, contexts, e2e_duration, args.slo_ttft, args.slo_tpot)
    if metrics is None:
        logger.warning("Failed to get metrics")
        return
    for ctx in contexts:
       if ctx.error:
            logger.warning(f"[{ctx.index}] ERROR: {ctx.error}")
       if not args.disable_warn_dismatch_output_len and ctx.max_tokens > 0 and abs(ctx.output_len - ctx.max_tokens) > 10:
            logger.warning(f"[{ctx.index}] Mismatched output length: expected {ctx.max_tokens}, got {ctx.output_len}")
    e2e_latency_p, ttft_p, tpot_p, tps_p = metrics.get_percentile([50, 90, 99])
    e2e_latency_avg, ttft_avg, tpot_avg, tps_avg = metrics.get_avg()

    prompt_len, gen_len = 0, 0
    if "llama-70b-completions_online_dataset" in args.dataset or "csv" in args.dataset:
        prompt_len = sum(sample[1] for sample in samples) / len(samples)
        gen_len = sum(sample[2] for sample in samples) / len(samples)
    elif args.sampling_policy == "fixed":
        prompt_len, gen_len = args.fixed_prompt_len, args.fixed_output_len
    elif args.sampling_policy == "normal":
        prompt_len, gen_len = args.prompt_len_mean, args.output_len_mean
    output = f"\n[BeginMetrics] {datetime.now().strftime('%m%d:%H-%M')}\n"
    output += f"log: {args.log_file}\n"
    output += f"model: {args.model}\n"
    output += f"sampling-policy: {args.sampling_policy}\n"
    output += f"sequence-length: {prompt_len}, {gen_len}\n"
    output += f"num-requests: {args.num_requests}\n"
    output += f"batch-size: {args.parallel}\n"
    output += f"e2e-latency(avg, P50, P90, P99): {e2e_latency_avg:0.2f}, {e2e_latency_p[0]:.2f}, {e2e_latency_p[1]:.2f}, {e2e_latency_p[2]:.2f}\n"
    output += f"ttft(avg, P50, P90, P99): {ttft_avg:.2f}, {ttft_p[0]:.2f}, {ttft_p[1]:.2f}, {ttft_p[2]:.2f}\n"
    output += f"tpot(avg, P50, P90, P99): {tpot_avg:.2f}, {tpot_p[0]:.3f}, {tpot_p[1]:.3f}, {tpot_p[2]:.3f}\n"
    output += f"tps(avg, P50, P90, P99): {tps_avg:.2f}, {tps_p[0]:.1f}, {tps_p[1]:.1f}, {tps_p[2]:.1f}\n"
    output += f"throughput: {metrics.input_tokens/e2e_duration:.2f}, {metrics.output_tokens/e2e_duration:.2f}, {(metrics.input_tokens+metrics.output_tokens)/e2e_duration:.2f}\n"
    output += f"rps: {len(contexts)/e2e_duration:.3f}\n"
    output += f"goodput-throughput: {metrics_good.output_tokens/e2e_duration:.2f}\n"
    output += f"goodput-rps: {len(metrics_good.ttft)/e2e_duration:.3f}\n"
    output += f"e2e_duration: {e2e_duration}\n"
    output += f"errors: {len(metrics.errors)}\n"
    output += f"raw-ttft: {metrics.ttft}\n"
    output += f"raw-tpot: {metrics.tpot}\n"
    output += f"[EndMetrics]\n"
    print(output)
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # LLM Server
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1, or http://localhost:8000/v2/models/ensemble")
    parser.add_argument("--backend", type=str, default="vllm", help="The backend e.g. vllm, trtllm, default is vllm")
    parser.add_argument("--api-key", type=str, help="The api key to call commercial inference API")
    parser.add_argument("--api-kind", type=str, default="completions", choices=["chat", "completions"], help="Can be: chat or completions(default)")
    # input parameters
    parser.add_argument("--model", type=str, help="The model name, if not set, call 'endpoint/models' to query")
    # test data sampling
    parser.add_argument("--sampling-policy", type=str, default="nature", choices=["nature", "fixed", "normal", "undefined", "order"])
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--fixed-prompt-len", type=int, default=3500)
    parser.add_argument("--fixed-output-len", type=int, default=500)
    parser.add_argument("--prompt-len-mean", type=int, default=550)
    parser.add_argument("--prompt-len-std", type=int, default=150)
    parser.add_argument("--output-len-mean", type=int, default=150)
    parser.add_argument("--output-len-std", type=int, default=20)
    # press test setting
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of prompts for benckmark.")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--dataset", type=str, help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--disable-stream", action="store_true", help="Disable stream mode")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Ignore EOS of the output")
    parser.add_argument("--disable-warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    parser.add_argument("--add-stream-usage", action="store_true", help="include stream usage in the request")
    # log
    parser.add_argument("--log-file", type=str, help="file to save log information")
    parser.add_argument("--record-raw-metrics", action="store_true", help="Dump raw metrics like TTFT or TPOT")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
    parser.add_argument("--max-sample-request-tokens", type=int, default=16384, help="set max sample requests tokens")
    parser.add_argument("--slo-ttft", type=float, default=2, help="the sla-ttft milliseconds used to calculate the goodput")
    parser.add_argument("--slo-tpot", type=float, default=0.08, help="the sla-ttft milliseconds used to calculate the goodput")
    parser.add_argument("--burstgpt-scale", type=float,  default=100,
                        help="Scale burstgpt trace, 100 means 100 times faster, use 1 by default")

    args = parser.parse_args()

    # set_ulimit: target_soft_limit=65535
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < 65535:
        try:
            resource.setrlimit(resource_type, (65535, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")
    main(args)

