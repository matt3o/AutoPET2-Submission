import torch
from pynvml import *

def get_gpu_usage(device:torch.device, used_memory_only=False, context=""):
    global logger
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device.index)
    info = nvmlDeviceGetMemoryInfo(h)
    usage = ""
    if used_memory_only:
        usage += '{} Device: {}\nused:  {:.0f} MiB\n'.format(context, device.index, info.used / (1024**2))
    else:
        usage += '{} Device: {}\ntotal: {:.0f} MiB\nfree:  {:.0f} MiB\nused:  {:.0f} MiB\n'.format(
            context, device.index, info.total/(1024**2), info.free / (1024**2), info.used / (1024**2))
        free, total_memory = torch.cuda.mem_get_info(device=device)
        usage += "torch device {} \ntotal_memory: {:.0f} MiB\nfree:{:.0f} MiB\n used: {} MiB\n".format(
            device.index, total_memory / (1024**2), free / (1024**2), (total_memory - free) / (1024**2))
    return usage


def print_gpu_usage(*args, **kwargs):
    print(get_gpu_usage(*args, **kwargs))
