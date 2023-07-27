# Common ways to call the script 


## For Training

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/20 -d /projects/mhadlich_segmentation/data/20g/data -c /local/work/mhadlich/cache -ta -e 400

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/22 -d /projects/mhadlich_segmentation/data/22/data -c /local/work/mhadlich/cache -e 400 --log

python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/87 -d /projects/mhadlich_segmentation/data/87/data -c /local/work/mhadlich/cache -ta -e 200 -f 10 --sw_batch_size 32 --scheduler PolynomialLR

Slurm run on IKIM:
srun --partition GPUampere --gpus 1 -J 86 --time 4-00:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SimpleInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/86 -d /projects/mhadlich_segmentation/data/86/data -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_crop_size '(192,192,256)'"



## For evaluation

python train.py -a --inferer=SlidingWindowInferer --disks --network dynunet --sigma 1 -o /tmp/output -d /tmp/data -c /tmp/cache -e 1 -t 1 --eval_only --save_nifti --model_weights data/18_checkpoint.pt --resume -ta

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/tmp -d /projects/mhadlich_segmentation/tmp -c /local/work/mhadlich/cache -ta -e 1 -t 1 --eval_only --save_nifti --model_weights '/projects/mhadlich_segmentation/data/30/checkpoint_epoch=30.pt' --resume


# to make sure you are on the correct server:
[[ -n "$CUDA_VISIBLE_DEVICES" ]] && echo yes


## ENV
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync;
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8;
CUPY_GPU_MEMORY_LIMIT="18%";

## Debugging options

PYTORCH_NO_CUDA_MEMORY_CACHING=1 