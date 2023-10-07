from __future__ import annotations

import glob
import logging
import os
import tempfile

import monai.transforms as mt
import nibabel as nib
import numpy as np
import torch
from monai.data import ArrayDataset, DataLoader, MetaTensor, create_test_image_3d, partition_dataset, set_track_meta
from monai.data.dataset import Dataset, PersistentDataset

NETWORK_INPUT_SHAPE = (1, 128, 128, 256)
NUM_IMAGES = 1

logger = logging.getLogger("sw_fastedit")
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
    print("### Run 1: Should trigger no warning")
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

        train_ds = Dataset(datalist, transform)

        train_ds2 = Dataset(datalist, transform)

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=1,
            batch_size=1,
            multiprocessing_context="spawn",
        )

        train_loader2 = DataLoader(
            train_ds2,
            shuffle=True,
            num_workers=1,
            batch_size=1,
        )
        set_track_meta(False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for x in train_loader:
            print(type(x["image"]))

        for x in train_loader2:
            print(type(x["image"]))

        print(type(transform(datalist[0])["image"]))

    print("### Run 2: Should trigger a warning for the first data loader")
    set_track_meta(False)
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

        train_ds = Dataset(datalist, transform)

        train_ds2 = Dataset(datalist, transform)

        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            num_workers=1,
            batch_size=1,
            multiprocessing_context="spawn",
        )

        train_loader2 = DataLoader(
            train_ds2,
            shuffle=True,
            num_workers=1,
            batch_size=1,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for x in train_loader:
            print(type(x["image"]))

        for x in train_loader2:
            print(type(x["image"]))

        print(type(transform(datalist[0])["image"]))
