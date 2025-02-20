import sys
import os
import argparse
import threading

from profile_server import ServerOnline, Config
from monitor_gpu.monitor import Monitor
 
def add_parser_arg(parser):
    parser.add_argument('--data_path', type=str, 
                        help='The path of the alpaca json file',
                        default="/preprocess_data/shareGPT.json")
    parser.add_argument('--model_path', type=str, 
                        help='The path of the tokenizer')
    parser.add_argument('--log_path', type=str, 
                        help='The path the log file to save',
                        default="./server_log.json")
    parser.add_argument("--detail_log_path", type=str, default="./detail_server_log_1.json",
                        help='The detail log path')
    parser.add_argument('--max_num_seqs', type=int, 
                        help='The maximum seqs batched',
                        default=256)
    parser.add_argument('--max_num_batched_tokens', type=int, 
                        help='The maximum tokens batched',
                        default=4096)

    parser.add_argument("--ignore_eos", action="store_true", default=False,
                        help="Ignore the eos")
    parser.add_argument("--max_tokens", type=int, 
                        help="The maximum tokens generated", default=128)
    parser.add_argument("--stream", action='store_true', default=False,
                        help="If the output is streaming or not, only for server mode")
    parser.add_argument("--qps", type=float, default=1.0,
                        help="Query Per Second, \
                        the parameter of Possion distribution to generate the query.")
    parser.add_argument("--host", type=str, default='localhost',
                        help="The server host")
    parser.add_argument("--port", type=int, default=17717,
                        help="The server port")

    #https://platform.openai.com/docs/api-reference/completions/create
    parser.add_argument("--temperature", type=int, default=0,
                        help="Ref to Openai api temperature")

    parser.add_argument('--seed', type=int, default=0,
                        help="The random seed of prompt set used in shuffle")

    parser.add_argument("--surplus_prompts_num", type=int, default=16384,
                        help="The total query num(surplus) to send")
    
    parser.add_argument("--use_burstgpt", action="store_true", default=False,
                        help="Using BurstGPT trace instead of using gamma and zipf")

    parser.add_argument("--burstgpt_path", type=str, 
                        help="BurstGPT trace path",
                        default="../data/BurstGPT_1.csv")
    
    parser.add_argument("--prompt_num", type=int,  default=500,
                        help="Prompt number, 500 by default")
    
    parser.add_argument("--conv_or_api", type=str,  default='conv',
                        help="Using BurstGPT Conv or API trace, use conv by default")
    
    parser.add_argument("--scale", type=float,  default=1,
                        help="Scale trace, 100 means 100 times faster, use 1 by default")

    parser.add_argument("--gpu_log_path", type=str, default='./logs/gpu_log/gpu_log.json',
                        help="GPU log loading path")
    
    parser.add_argument("--vllm_log_path", type=str, default='',
                        help="vLLM log loading path")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()

    server_config = dict()
    prompt_config = dict()
    server_config['stream'] = args.stream
    server_config['ignore_eos'] = args.ignore_eos
    server_config['qps'] = args.qps
    server_config['host'] = args.host
    server_config['port'] = args.port
    server_config['temperature'] = args.temperature
    server_config['max_tokens'] = args.max_tokens

    prompt_config['seed'] = args.seed
    prompt_config['surplus_prompts_num'] = args.surplus_prompts_num
    prompt_config['use_burstgpt'] = args.use_burstgpt
    prompt_config['burstgpt_path'] = args.burstgpt_path
    prompt_config['conv_or_api'] = args.conv_or_api
    prompt_config['scale'] = args.scale
    prompt_config['prompt_num'] = args.prompt_num

    if len(args.vllm_log_path) != 0:
        with open("/root/lss/BurstGPT_test/example/logs/vllm_log/vllm_log.csv", "w") as f:
            f.write("timestamp,prompt_throughput,generation_throughput,num_running,num_swapped,"
                    "num_pending,gpu_kvcache_usage,cpu_kvcache_usage\n")
            
    print(prompt_config)
    # monitor GPU
    monitor = Monitor(args.gpu_log_path)
    config = Config(server_config=server_config, prompt_config=prompt_config)
    server = ServerOnline(model_path=args.model_path,
                           data_path=args.data_path,
                           backend="vllm",
                           log_path=args.log_path,
                           config=config,
                           detail_log_path=args.detail_log_path,
                           monitor=monitor,
                           vllm_log_path=args.vllm_log_path
                           )
    # Monitor GPU while sending query
    try:
        monitor_thread = threading.Thread(target=Monitor.start_monitor, args=(monitor, 0.5,))
        server_thread = threading.Thread(target=server.start_profile)
        
        monitor_thread.start()
        server_thread.start()
        server_thread.join()
        
    except :
        print("Stop")
        monitor.save_gpu_log()
        server.save_log()
        if len(args.vllm_log_path) != 0:
            os.system(f"mv ./logs/vllm_log/vllm_log.csv {args.vllm_log_path}")
        print("Exit!")