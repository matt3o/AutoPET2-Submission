# Common ways to call the script 


## For Training

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/20 -d /projects/mhadlich_segmentation/data/20g/data -c /local/work/mhadlich/cache -ta -e 400 --model_weights /projects/mhadlich_segmentation/data/18/checkpoint.pt --resume --log

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/22 -d /projects/mhadlich_segmentation/data/22/data -c /local/work/mhadlich/cache -e 400 --log


## For evaluation

python train.py -a --inferer=SlidingWindowInferer --disks --network dynunet --sigma 1 -o /tmp/output -d /tmp/data -c /tmp/cache -e 1 -t 1 --eval_only --save_nifti --model_weights data/18_checkpoint.pt --resume -ta

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/tmp -d /projects/mhadlich_segmentation/tmp -c /local/work/mhadlich/cache -ta -e 1 -t 1 --eval_only --save_nifti --model_weights /projects/mhadlich_segmentation/data/18/checkpoint.pt --resume


# to make sure you are on the correct server:
[[ -n "$CUDA_VISIBLE_DEVICES" ]] && echo yes


## ENV
PYTORCH_CUDA_ALLOC_CONF=cudaMallocAsync;
CUPY_GPU_MEMORY_LIMIT="18%";