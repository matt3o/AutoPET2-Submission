# Sliding Window Based Interactive Segmentation of Volumetric Medical Images

An extension / rework of the DeepEdit code. Most work has been put into moving the transform to torch and thus on the GPU while preventing the common OOMs with MONAI. Also cupy based distance transforms have been integrated to remove the old scipy based ones. More specifically the scipy cdt distance transform has been replaced by an edt one.

This code was exracted from MONAI, reworked by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology (zdravko.marinov@kit.edu) to make it's way into my master thesis, which is this repository.
B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology (matthiashadlich@posteo.de)



## Common ways to call the script 

### For Training

python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/20  -c /local/work/mhadlich/cache -ta -e 400

python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/87 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_sw_batch_size 32 --scheduler PolynomialLR


### For evaluation

python train.py -a --inferer=SlidingWindowInferer --disks --network dynunet --sigma 1 -o /tmp/output -d /tmp/data -c /tmp/cache -e 1 -t 1 --eval_only --save_nifti --resume_from data/18_checkpoint.pt -ta

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/tmp -d /projects/mhadlich_segmentation/tmp -c /local/work/mhadlich/cache -ta -e 1 -t 1 --eval_only --save_nifti --resume_from '/projects/mhadlich_segmentation/data/30/checkpoint_epoch=30.pt'

### ENV

Recommended: PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8;

No longer necessary: CUPY_GPU_MEMORY_LIMIT="18%";
Not recommended: PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync;

### Debugging options

PYTORCH_NO_CUDA_MEMORY_CACHING=1 
-> Care it slows down the network extremly


## IKIM

Slurm run on IKIM

srun -X --partition GPUampere --gpus 1 -J 144 --time 8-00:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/144 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_sw_batch_size 8 --scheduler CosineAnnealingLR --network ultradynunet"

srun --partition GPUampere --gpus 1 -J 86 --time 4-00:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SimpleInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/86 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_crop_size '(192,192,256)'"

srun -X --partition GPUampere --gpus 1 -J tmp --time 0-03:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/eval -c /local/work/mhadlich/cache -ta --val_sw_batch_size 8 --dont_check_output_dir --resume_from /projects/mhadlich_segmentation/data/104/checkpoint.pt --eval_only -t 10"

To make sure you are on the correct server:
[[ -n "$CUDA_VISIBLE_DEVICES" ]] && echo CUDA activated
