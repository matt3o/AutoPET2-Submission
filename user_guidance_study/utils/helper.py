import torch
from pynvml import *
import gc
from monai.data.meta_tensor import MetaTensor

import functools  
import time
from datetime import datetime

import os

def get_actual_cuda_index_of_device(device:torch.device):
    try:
        cuda_indexes = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    except KeyError:
        return int(device.index)
    return int(cuda_indexes[device.index])

def get_gpu_usage(device:torch.device, used_memory_only=False, context="", csv_format=False):
    global logger
    nvmlInit()
    cuda_index = get_actual_cuda_index_of_device(device)
    h = nvmlDeviceGetHandleByIndex(cuda_index)
    info = nvmlDeviceGetMemoryInfo(h)
    nv_total, nv_free, nv_used = info.total / (1024**2), info.free / (1024**2), info.used / (1024**2)
    usage = ""
    utilization = torch.cuda.utilization(device)
    t_free, t_total = [i / (1024**2) for i in torch.cuda.mem_get_info(device=device)]

    t_used = t_total - t_free
    used_not_by_torch = nv_used - t_used
    amount_of_tensors = print_amount_of_tensors()
    if csv_format and used_memory_only:
        raise NotImplemented

    if csv_format:
        header = "device,context,time,utilization,total memory (MB),free memory (MB),used memory (MB),memory not used by torch (MB),amount_of_tensors"
        usage += '{},{},{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}'.format(
            cuda_index, context, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), utilization, nv_total, nv_free, nv_used, used_not_by_torch, amount_of_tensors)
        return (header, usage)
    else:
        if used_memory_only:
            usage += '{} Device: {} --- used:  {:.0f} MB\n'.format(context, cuda_index, nv_used)
        else:
            usage += '{},{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{}'.format(
                cuda_index, context, utilization, nv_total, nv_free, nv_used, used_not_by_torch, amount_of_tensors)
    return usage


def print_gpu_usage(*args, **kwargs):
    print(get_gpu_usage(*args, **kwargs))

def print_tensor_gpu_usage(a:torch.Tensor):
    print("Tensor GPU memory: {} MB".format(a.element_size() * a.nelement() / (1024**2)))


def print_all_tensor_gpu_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def print_amount_of_tensors():
    counter = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                counter += 1
            # if torch.is_tensor(obj) and torch.is_cuda(obj):
            #     counter += 1
            # elif (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and torch.is_cuda(obj.data):
            #     counter += 1
        except:
            pass
    print(f"#################################### Amount of tensors: {counter}")


def get_total_size_of_all_tensors(data):
    size = 0
    if type(data) == dict:
        for key in data:
            size += get_total_size_of_all_tensors(data[key])
    elif type(data) == list:
        for element in data:
            size += get_total_size_of_all_tensors(element)
    elif type(data) == torch.Tensor or type(data) == MetaTensor:
        size += data.element_size() * data.nelement()

    return size

def describe(t:torch.Tensor):
    return "mean: {} \nmin: {}\nmax: {} \ndtype: {} \ndevice: {}".format(torch.mean(t), torch.min(t), torch.max(t), t.dtype, t.device)

def describe_batch_data(batchdata: dict, total_size_only=False):
    batch_data_string = ""
    if total_size_only:
        batch_data_string += f"Total size of all tensors in batch data: {get_total_size_of_all_tensors(batchdata)/ (1024**2)} MB"
    else:
        batch_data_string += f"Type of batch data: {type(batchdata)}"
        for key in batchdata:
            if type(batchdata[key]) == torch.Tensor:
                batch_data_string += f"{key} size: {batchdata[key].size()} size in MB: {batchdata[key].element_size() * batchdata[key].nelement() / (1024**2)}MB"
            elif type(batchdata[key]) == dict:
                for key2 in batchdata[key]:
                    if type(batchdata[key][key2]) == torch.Tensor:
                        batch_data_string += f"{key}/{key2} size: {batchdata[key][key2].size()} size in MB: {batchdata[key][key2].element_size() * batchdata[key][key2].nelement() / (1024**2)}MB"
                    else:
                        batch_data_string += f"{key}/{key2}: {batchdata[key][key2]}"
            else:
                raise UserWarning()
    return batch_data_string

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}() took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
