import argparse
import time
import os

from typing import Tuple
import aiohttp
import asyncio
import time
import json

from monitor_gpu.monitor import Monitor

async def vllm_inference_call_server(prompt, in_num, out_num, sampled_in_num, sampled_out_num, sleep_time, config, logger, event_id) -> dict:
    await asyncio.sleep(sleep_time)
    monitor = Monitor("./logs/test.json")
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)
    print(f"[INFO] Start {event_id}, after sleep: {sleep_time}")
    async with aiohttp.ClientSession(timeout=timeout) as session:
        generation_input = {
            "prompt": prompt,
            "stream": config.server_config['stream'],
            "ignore_eos": False,
            "max_tokens": int(out_num),
            "temperature": config.server_config['temperature'],
        }
        first_chunk_time = 0
        start_time = time.perf_counter()
        async with session.post(
            f"http://{config.server_config['host']}:{config.server_config['port']}/generate", json=generation_input
        ) as resp:
            
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return None, None, None
            
            gpu_info = monitor.get_current_info()
            print(gpu_info)


            if config.server_config['stream']:
                buffer = b""
                first_chunk_received = False
                async for chunk in resp.content.iter_any():
                    buffer += chunk

                    # If this is the first chunk, record the time taken
                    if not first_chunk_received:
                        first_chunk_time = time.perf_counter() - start_time
                        first_chunk_received = True

                    while b"\0" in buffer:  # Split by null character
                        json_str, buffer = buffer.split(b"\0", 1)
                output = json.loads(json_str.decode("utf-8"))  # Decode JSON

            else:
                output = await resp.json()
            
            end_time = time.perf_counter()
            total_chunk_time = end_time - start_time

            # should counter the output token length after gather all the outputs
    # logger.tick_end(event_id, time.perf_counter())

    save_query_json = {"event_id":event_id, 
                       "out_len":len(output["text"][0]), 
                       "out_len_expected": int(out_num), 
                       "in_len":int(in_num), 
                       "sampled_in_num": int(sampled_in_num), 
                       "sampled_out_len":int(sampled_out_num), 
                       "first_chunk_time":first_chunk_time, 
                       "total_chunk_time":total_chunk_time, 
                       "record_time":time.perf_counter(), 
                       "GPU_used_mem_percent":float(gpu_info.used_mem/gpu_info.total_mem),
                       "GPU_used_mem":int(gpu_info.used_mem), 
                       "GPU_utilization":int(gpu_info.utilization),
                       "GPU_temperature":int(gpu_info.temperature)}
    logger.log_kv(f"{event_id}", save_query_json)
    # with open(logger.log_path, "a") as f:
    #     f.write("\n")
    #     json.dump(save_query_json, f)