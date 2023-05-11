import torch
from pynvml import *
import gc
from monai.data.meta_tensor import MetaTensor

def get_gpu_usage(device:torch.device, used_memory_only=False, context=""):
    global logger
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device.index)
    info = nvmlDeviceGetMemoryInfo(h)
    usage = ""
    if used_memory_only:
        usage += '{} Device: {} --- used:  {:.0f} MiB\n'.format(context, device.index, info.used / (1024**2))
    else:
        usage += '{} Device: {}\ntotal: {:.0f} MiB\nfree:  {:.0f} MiB\nused:  {:.0f} MiB\n'.format(
            context, device.index, info.total/(1024**2), info.free / (1024**2), info.used / (1024**2))
        free, total_memory = torch.cuda.mem_get_info(device=device)
        usage += "torch device {} \ntotal_memory: {:.0f} MiB\nfree:{:.0f} MiB\n used: {} MiB\n".format(
            device.index, total_memory / (1024**2), free / (1024**2), (total_memory - free) / (1024**2))
    return usage


def print_gpu_usage(*args, **kwargs):
    print(get_gpu_usage(*args, **kwargs))

def print_tensor_gpu_usage(a:torch.Tensor):
    print("Tensor GPU memory: {} Mib".format(a.element_size() * a.nelement() / (1024**2)))


def print_all_tensor_gpu_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def print_amount_of_tensors_on_gpu():
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
    print(f"#################################### No of GPU tensors: {counter}")


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