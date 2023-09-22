
Link to the weights of the submission: https://bwsyncandshare.kit.edu/s/Jg3K6Y35jXiBzpK

# Sliding Window Based Interactive Segmentation of Volumetric Medical Images

An extension / rework of the DeepEdit code. Most work has been put into moving the transform to torch and thus on the GPU while preventing the common OOMs with MONAI. Also cupy based distance transforms have been integrated to remove the old scipy based ones. More specifically the scipy cdt distance transform has been replaced by an edt one.

The entire project is a Python, so I can be installed with `pip install -e`. This was added mostly for the MONAILabel Plugin.

Training with 10 clicks on (224,224,224) patches with validation on full volumes finishes in under a week on a single Nvidia A6000 50Gb.

This code was exracted from MONAI, reworked by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology (zdravko.marinov@kit.edu) to make it's way into my master thesis, which is this repository.

2023, B.Sc. Matthias Hadlich, Karlsuhe Institute of Technology (matthiashadlich@posteo.de)

A paper on the results from this repository will follow as well. A user study will be done to evaluate the performance of humans vs the robot clicks.

**Important**: This code is only tested on AutoPET, it will no longer work for MSD Spleen. Also the support 2D images has been dropped - at least I think it won't work any longer..


## Performance of this repo on AutoPET

With 10 clicks / guidance during validation: 0.8707 dice on the full volumes after 400 epochs of training

Without clicks / guidance during validation: 0.7304 dice on the full volumes after 400 epochs of training

The `SlidingWindowInferer` does not only beat the `SimpleInferer` in terms of comfort (volumes of any size can be used), but also in performance.
When comparing both, on a `val_crop_size` of `(192,192,256)`, the sliding window approach yield 0.8383 vs 0.8102 on the SimpleInferer.

## Notes about the GPU usage

This code has initially run on 11 Gb GPU and should still run on 24Gb GPUs if you set `train_crop_size` and `val_crop_size` low enough (`--val_crop_size='(128,128,128)'`). train_crop_size has to be way lower since during training the gradients have to calculated and backpropagated.

Most of this code runs on magic since MONAI is leaking memory in the same way Pytorch does. For more details look into https://github.com/Project-MONAI/MONAI/issues/6626 - the described problem is based on this code. In essence: Usually the problem is not that torch uses too much memory but rather the garbage collector does not clean the MONAI / torch objects often enough so that pointers to GPU memory remain in the memory for too long and hog GPU memory which in theory would already be free for use. Thus we need to encourage the gc to collect more often which is done with the GarbageCollector Handler.

If you run into unexpected OOMs (which can still happen sadly): 

- Increase the GarbageCollector collection steps
- Don't move any data on the GPU during the pre transforms (spent countless days with that..)
- Try `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8;` which autoclears the torch GPU memory if increases over 80%
- There are a ton of debbuging options implemented in this repository, starting at the GPU_Thread over transform that print the GPU (Note that the dataloader can spawn in a different process and thus use GPU memory independend of the main process. The memory won't pop up in the `torch.cuda.memory_summary()`)

When desperate enough try: `PYTORCH_NO_CUDA_MEMORY_CACHING=1` -> Care it slows down the network extremly. If it runs without caching then you have found a memory leak.

Not working I think:
- In the past I restricted cupy with `CUPY_GPU_MEMORY_LIMIT="18%";`, however this appears to pin the cupy and thus overall increase the memory cupy used
- Not recommended: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync;`


## Common ways to call the script 

### For Training

```bash
python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/20  -c /local/work/mhadlich/cache -ta -e 400

python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/87 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_sw_batch_size 32 --scheduler PolynomialLR
```


### For evaluation

```bash
python train.py -a --inferer=SlidingWindowInferer --disks --network dynunet --sigma 1 -o /tmp/output -d /tmp/data -c /tmp/cache -e 1 -t 1 --eval_only --save_nifti --resume_from data/18_checkpoint.pt -ta

[[ -n "$CUDA_VISIBLE_DEVICES" ]] && python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/tmp -d /projects/mhadlich_segmentation/tmp -c /local/work/mhadlich/cache -ta -e 1 -t 1 --eval_only --save_nifti --resume_from '/projects/mhadlich_segmentation/data/30/checkpoint_epoch=30.pt'
```


## IKIM

Slurm run on IKIM

```bash
srun -X --partition GPUampere --gpus 1 -J 144 --time 8-00:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/144 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_sw_batch_size 8 --scheduler CosineAnnealingLR --network ultradynunet"

srun --partition GPUampere --gpus 1 -J 86 --time 4-00:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SimpleInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/86 -c /local/work/mhadlich/cache -ta -e 200 -f 10 --val_crop_size '(192,192,256)'"

srun -X --partition GPUampere --gpus 1 -J tmp --time 0-03:00:00 bash -c "python train.py -a --disks --sigma 1 --inferer SlidingWindowInferer -i /projects/mhadlich_segmentation/AutoPET/AutoPET -o /projects/mhadlich_segmentation/data/eval -c /local/work/mhadlich/cache -ta --val_sw_batch_size 8 --dont_check_output_dir --resume_from /projects/mhadlich_segmentation/data/104/checkpoint.pt --eval_only -t 10"
```

To make sure you are on the correct server:
`[[ -n "$CUDA_VISIBLE_DEVICES" ]] && echo CUDA activated`


## Evaluation with the MONAILabel Plugin

TODO: Add info here

