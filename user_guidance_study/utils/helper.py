import torch
from pynvml import *
import gc
from monai.data.meta_tensor import MetaTensor

import functools  
import time
from datetime import datetime

import os

import logging
logger = logging.getLogger("interactive_segmentation")


def get_actual_cuda_index_of_device(device:torch.device):
    try:
        cuda_indexes = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    except KeyError:
        return int(device.index)
    return int(cuda_indexes[device.index])

def gpu_usage(device:torch.device, used_memory_only=False):
    # empty the cache first
    torch.cuda.empty_cache()
    nvmlInit()
    cuda_index = get_actual_cuda_index_of_device(device)
    h = nvmlDeviceGetHandleByIndex(cuda_index)
    info = nvmlDeviceGetMemoryInfo(h)
    nv_total, nv_free, nv_used = info.total / (1024**2), info.free / (1024**2), info.used / (1024**2)
    utilization = torch.cuda.utilization(device)
    t_free, t_total = [i / (1024**2) for i in torch.cuda.mem_get_info(device=device)]

    t_used = t_total - t_free
    used_not_by_torch = nv_used - t_used

    if not used_memory_only:
        return cuda_index, utilization, nv_total, nv_free, nv_used, used_not_by_torch
    else:
        return nv_used


def get_gpu_usage(device:torch.device, used_memory_only=False, context="", csv_format=False):
    cuda_index, utilization, nv_total, nv_free, nv_used, used_not_by_torch = gpu_usage(device=device)
    usage = ""

    if csv_format and used_memory_only:
        raise NotImplemented

    if csv_format:
        header = "device,context,time,utilization,total memory (MB),free memory (MB),used memory (MB),memory not used by torch (MB)"
        usage += '{},{},{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}'.format(
            cuda_index, context, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), utilization, nv_total, nv_free, nv_used, used_not_by_torch)
        return (header, usage)
    else:
        if used_memory_only:
            usage += '{} Device: {} --- used:  {:.0f} MB'.format(context, cuda_index, nv_used)
        else:
            usage += '{},{},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}'.format(
                cuda_index, context, utilization, nv_total, nv_free, nv_used, used_not_by_torch)
    return usage


def print_gpu_usage(*args, **kwargs):
    logger.info(get_gpu_usage(*args, **kwargs))

def print_tensor_gpu_usage(a:torch.Tensor):
    logger.info("Tensor GPU memory: {} MB".format(a.element_size() * a.nelement() / (1024**2)))


def print_all_tensor_gpu_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logger.info(type(obj), obj.size())
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
    return counter


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
        batch_data_string += f"Total size of all tensors in batch data: {get_total_size_of_all_tensors(batchdata)/ (1024**2)} MB\n"
    else:
        batch_data_string += f"Type of batch data: {type(batchdata)}\n"
        for key in batchdata:
            if type(batchdata[key]) == torch.Tensor:
                batch_data_string += (
                    f"- {key}(Tensor) size: {batchdata[key].size()} "
                    f"size in MB: {batchdata[key].element_size() * batchdata[key].nelement() / (1024**2)}MB "
                    f"device: {batchdata[key].device} "
                    f"dtype: {batchdata[key].dtype} \n"
                )
            elif type(batchdata[key]) == MetaTensor:
                batch_data_string += (
                    f"- {key}(MetaTensor) size: {batchdata[key].size()} "
                    f"size in MB: {batchdata[key].element_size() * batchdata[key].nelement() / (1024**2)}MB "
                    f"device: {batchdata[key].device} "
                    f"dtype: {batchdata[key].dtype} \n"
                )
                batch_data_string += f"  Meta: {batchdata[key].meta}\n"""
            elif type(batchdata[key]) == dict:
                batch_data_string += f"- {key}(dict)\n"
                for key2 in batchdata[key]:
                    if type(batchdata[key][key2]) == torch.Tensor or type(batchdata[key][key2]) == MetaTensor:
                        batch_data_string += (
                            f"    - {key}/{key2}(Tensor/MetaTensor) "
                            f"size: {batchdata[key][key2].size()} "
                            f"size in MB: {batchdata[key][key2].element_size() * batchdata[key][key2].nelement() / (1024**2)}MB "
                            f"device: {batchdata[key][key2].device} "
                            f"dtype: {batchdata[key][key2].dtype}\n"
                        )
                    else:
                        batch_data_string += f"    - {key}/{key2}: {batchdata[key][key2]} \n"
            elif type(batchdata[key]) == list:
                batch_data_string += f"- {key}(list)\n"
                for item in batchdata[key]:
                    batch_data_string += f"    - {item}\n"
            else:
                batch_data_string += f"- {key}({type(batchdata[key])})\n"
                # logger.error(f"Unknown datatype: {type(batchdata[key])}")
                # raise UserWarning()
    return batch_data_string

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        try:
            device = args[0].device
        except:
            device = None
        
        if device is not None:
            gpu1 = gpu_usage(device=device, used_memory_only=True)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        if device is not None:
            gpu2 = gpu_usage(device=device, used_memory_only=True)
        total_time = end_time - start_time
        if device is not None:
            logger.info(f'Function {func.__qualname__}() took {total_time:.4f} seconds and reserved {(gpu2 - gpu1) / 1024**2:.1f} MB GPU memory')
        else:
            logger.info(f'Function {func.__qualname__}() took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def get_git_information():
    stream = os.popen('git branch;git rev-parse HEAD')
    git_info = stream.read()
    return git_info