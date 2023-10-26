# AutoPET 2 submission

We are one of the top teams in the AutoPET II challenge! I am happy to answer any question, just write me an Email.

Authors: Matthias Hadlich, Zdravko Marinov, Rainer Stiefelhagen
Link to the Paper https://arxiv.org/abs/2309.12114

Link to the weights of the submission: https://bwsyncandshare.kit.edu/s/Jg3K6Y35jXiBzpK


# Sliding Window-based Interactive Segmentation of Volumetric Medical Images

An extension / rework of the DeepEdit code. Most work has been put into moving the transform to torch and thus on the GPU while preventing the common OOMs with MONAI. Also cupy based distance transforms have been integrated to remove the old scipy based ones. 
A lot of the improvements from this code have been integrated into MONAI 1.3.0.

This code was exracted from MONAI, reworked by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology (zdravko.marinov@kit.edu) to make it's way into my master thesis, which is this repository. 2023, M. Sc. Matthias Hadlich, Karlsuhe Institute of Technology (matthiashadlich@posteo.de)

**Important**: This code is only tested on 3D PET images of AutoPET(II). 2D images are not suppported, I think the code has to be adapted for that.


## Performance of this repo on AutoPET

Training for 200 epochs with 10 clicks on (224,224,224) patches with validation on full volumes finishes in under a week on a single Nvidia A6000 50Gb.
With 10 clicks / guidance during validation: 0.8715 dice on the full volumes after 800 epochs of training.
Without clicks / guidance during validation: 0.7407 dice on the full volumes after 400 epochs of training.

The `SlidingWindowInferer` does not only beat the `SimpleInferer` in terms of comfort (volumes of any size can be used), but also in terms of the Dice score. 
When comparing both, on a `val_crop_size` of `(192,192,256)`, the sliding window approach yields a Dice of 0.8383 vs 0.8102 on the SimpleInferer.

## Full deep learning workflow

### Training (image and label)

Use the `train.py` file for that. Use the `--resume_from` flag to resume training from an aborted previous experiment. Example usage:

`python train.py -a -i /projects/mhadlich_segmentation/AutoPET/AutoPET --dataset AutoPET -o /projects/mhadlich_segmentation/data/20  -c /local/work/mhadlich/cache -ta -e 400`

### Evaluation

Use the `train.py` file for that and only add the `--eval_only` flag. The network will only run the evaluator which finishes after one epoch. Evaluation will use the images and the label and thus print a metric at the end.
Use the `--resume_from` flag to load previous weights.
Use `--save_pred` to save the resulting predictions.

### Testing (image only)

Use the `test.py` file for running. Example usage on the AutoPET test mha files:

`python test.py -i /input/images/pet/ -o /output/images/automated-petct-lesion-segmentation/  --non_interactive -a --resume_from checkpoint.pt -ta --dataset AutoPET2_Challenge --dont_check_output_dir --no_log --sw_overlap 0.75 --no_data --val_sw_batch_size 8`

Also check out the Docker file for testing, it it configured to run on the AutoPET2 challenge files.

### monailabel

There are multiple steps involved to get this to run.

Optional: Create a new conda environment
1) Install monailabel via `pip install monailabel`.
2) Install the dependencies of this repository with `pip install -r requirements.txt`, then install this repository as a package via `pip install -e`. Hopefully this step can be removed in the future when the code is integrated into MONAI.
3) Download the radiology sample app `monailabel apps --download --name radiology --output .`
    (Alternative: Download the entire monailabel repo and just launch monailabel from there)
4) Copy the files from the repo under `monailabel/` to `radiology/lib/` and into the according folders `infers/` and `configs/`.
5) Download the weights from https://bwsyncandshare.kit.edu/s/Yky4x6PQbtxLj2H , rename it to `pretrained_sw_fastedit.pt` and put them into the (new) folder `radiology/model/`. This model was pretrained on tumor-only AutoPET volumes.
6) Make sure your images follow the monailabel convention, so e.g. all Nifti files in one folder `imagesTs`.

You can then run the model with (adapt the studies path where the images lie):

`monailabel start_server --app radiology --studies ../imagesTs --conf models sw_fastedit`


### Computing statistics on the monailabel results

Use compute_metrics.py for that. Example usage:

`python compute_metrics.py -l /projects/mhadlich_segmentation/AutoPET/AutoPET/labelsTs/ -p /projects/mhadlich_segmentation/user_study/baseline -o eval/
`

### Docker

For the AutoPET II challenge this code has been dockerized. For details check out `build.sh`, `export.sh`, `test_autopet.sh`, `test.sh`.

### Available Datasets

- AutoPET: Should not be used, this was our own remodelled tumor-only dataset. Consists of images in the folder imagesTs, labelsTs, imagesTr, labelsTr
- AutoPET II: Default for the AutoPET NIFTI structure, supply a link to the FDG-PET-CT-Lesions folder to start
- AutoPET2_Challenge: AutoPET II challenge mode, loads the mha files. Look into test.sh for details on how to call it
- HECKTOR: Default for the HECKTOR NIFTI dataset, so supply the HECKTOR folder with the two subfolders `hecktor2022_training` and `hecktor2022_testing`
- MSD Spleen: Untested, still exists for legacy reasons


## Notes about the GPU usage

This code has initially run on 11 Gb GPU and should still run on 24Gb GPUs if you set `train_crop_size` (and maybe also `val_crop_size`) low enough (e.g., `--train_crop_size='(128,128,128)'`).

Most of this code runs on magic since MONAI is leaking memory in the same way Pytorch does. For more details look into https://github.com/Project-MONAI/MONAI/issues/6626 - the described problem is based on this code. In essence: Usually the problem is not that torch uses too much memory but rather the garbage collector does not clean the MONAI / torch objects often enough so that pointers to GPU memory remain in the memory for too long and hog GPU memory which in theory would already be free for use. Thus we need to encourage the gc to collect more often which is done with the GarbageCollector Handler.

If you run into unexpected OOMs (which can still happen sadly), try to: 

- Increase the GarbageCollector collection steps
- Don't move any data on the GPU during the pre transforms
- There are a ton of debbuging options implemented in this repository, starting at the GPU_Thread and transforms that print the GPU memory usage (Note that the dataloader can spawn in a different process and thus use GPU memory independend of the main process. The memory won't pop up in the `torch.cuda.memory_summary()`)
- Manually set `gpu_size` to small
- Send me an Email and I'll try to help (matthiashadlichatyahoo.de)



Not working I think:
- When desperate enough try: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
- In the past I restricted cupy with `CUPY_GPU_MEMORY_LIMIT="18%";`, however this appears to pin the cupy and thus overall increase the memory cupy used
- Not recommended: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync;`, breaks with cupy
