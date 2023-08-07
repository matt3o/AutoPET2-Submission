import glob
import os
import logging
import tempfile

import numpy as np
import monai.transforms as mt
import torch
from monai.data import ArrayDataset, DataLoader, MetaTensor
from monai.data.dataset import Dataset, PersistentDataset

from monai.data import create_test_image_3d, partition_dataset
from monai.data import set_track_meta

import nibabel as nib

NETWORK_INPUT_SHAPE = (1, 128, 128, 256)
NUM_IMAGES = 1

logger = logging.getLogger("sw_interactive_segmentation")
if logger.hasHandlers():
    logger.handlers.clear()
logger.propagate = False
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
# (%(name)s)
formatter = logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d][%(levelname)s] %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


if __name__ == "__main__":
    #    location = "/projects/mhadlich_segmentation/AutoPET/AutoPET"
    #    all_images = sorted(glob.glob(os.path.join(location, "imagesTr", "*.nii.gz")))
    #    all_labels = sorted(glob.glob(os.path.join(location, "labelsTr", "*.nii.gz")))
    #    datalist = [
    #        {"image": image_name, "label": label_name}
    #        for image_name, label_name in zip(all_images, all_labels)
    #    ]  # if image_name not in bad_images]
    #

    # if have multiple nodes, set random seed to generate same random data for every node
    np.random.seed(seed=0)
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"generating synthetic data to {tmpdirname} (this may take a while)")
        for i in range(1):
            pred, label = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1, noise_max=0.5)
            n = nib.Nifti1Image(pred, np.eye(4))
            nib.save(n, os.path.join(tmpdirname, f"pred{i:d}.nii.gz"))
            n = nib.Nifti1Image(label, np.eye(4))
            nib.save(n, os.path.join(tmpdirname, f"label{i:d}.nii.gz"))
        print(os.path.join(str(tmpdirname), "pred*.nii.gz"))
        images = sorted(glob.glob(os.path.join(str(tmpdirname), "pred*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(str(tmpdirname), "label*.nii.gz")))
        datalist = [{"image": image, "label": label} for image, label in zip(images, labels)]

        #        datalist = datalist[0:1]

        device = "cuda"

        transform = mt.Compose(
            [
                mt.LoadImaged(
                    keys="image",
                    reader="ITKReader",
                    image_only=False,
                    simple_keys=True,
                ),
            ]
        )

        # train_ds = Dataset(datalist, transform)

        train_ds = Dataset(datalist, transform)  # , cache_dir='/tmp/cache/'

        train_ds2 = Dataset(datalist, transform)  # , cache_dir='/tmp/cache/'

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=1,
            batch_size=1,
            multiprocessing_context="spawn",  # persistent_workers=True,
        )

        train_loader2 = DataLoader(
            train_ds2,
            shuffle=True,
            num_workers=1,
            batch_size=1,  # , multiprocessing_context='spawn', #persistent_workers=True,
        )
        set_track_meta(False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for x in train_loader:
            print(type(x["image"]))

        for x in train_loader2:
            print(type(x["image"]))

        print(type(transform(datalist[0])["image"]))
