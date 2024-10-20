
import pynvml
import time
import argparse
import json

from profile_server import Logger

class Gpu_info:
    def __init__(self, total_mem=0, used_mem=0, utilization=0, temperature=0) -> None:
        self.total_mem = total_mem 
        self.used_mem = used_mem 
        self.utilization = utilization
        self.temperature = temperature

    def update(self, total_mem, used_mem, utilization, temperature) -> None:
        self.total_mem = total_mem 
        self.used_mem = used_mem 
        self.utilization = utilization
        self.temperature = temperature

    def __str__(self):
        return  f"Memory: used/total = {self.used_mem}/{self.total_mem}GB \t Utilization: {self.utilization}% \t Temperature: {self.temperature}C \t"           

class Monitor:
    def __init__(self, gpu_log_path) -> None:
        pynvml.nvmlInit()
        self.num_device = pynvml.nvmlDeviceGetCount()
        self.peak_mem_list = [0 for _ in range(self.num_device)]
        self.exit = False
        self.gpu_log_path = gpu_log_path
        self.gpu_info = Gpu_info()
        self.gpu_log_json = dict()

    def get_current_info(self) -> Gpu_info:
        for idx in range(self.num_device):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = int(mem_info.total / 1024 / 1024)
            used_mem = int(mem_info.used / 1024 / 1024)
            self.peak_mem_list[idx] = max(self.peak_mem_list[idx], used_mem)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        self.gpu_info.update(total_mem, used_mem, util, temp)
        return self.gpu_info

    def start_monitor(self, monitor_interval):
        monitor_id = 0
        while True:
            self.get_current_info()
            time.sleep(monitor_interval)
            print(self.gpu_info)
            current_log_json = {
                "monitor_id":monitor_id,  
                "record_time":time.perf_counter(), 
                "GPU_used_mem":int(f"{self.gpu_info.used_mem}"),
                "GPU_total_mem":int(f"{self.gpu_info.total_mem}"),
                "GPU_utilization":int(f"{self.gpu_info.utilization}"),
                "GPU_temperature":int(self.gpu_info.temperature)
            }
            self.gpu_log_json[f'{monitor_id}'] = current_log_json
            monitor_id += 1


    def save_gpu_log(self) -> None:
        print("[INFO]Saving gpu log data")
        with open(self.gpu_log_path, "w") as f:
            json.dump(self.gpu_log_json, f)

def monitor_script():
    parser = argparse.ArgumentParser(description="Monitor GPU tools", add_help=True)
    parser.add_argument("--interval", type=float, default=1.0, help="second to wait between updates")
    args = parser.parse_args()
    interval = args.interval
    print(f"update interval: {interval:.1f}s")
    monitor = Monitor()

    try:
        while True:
            gpu_info = monitor.get_current_info()
            print(gpu_info)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    monitor_script()