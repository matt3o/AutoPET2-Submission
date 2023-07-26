from __future__ import annotations

import glob
import logging
import os
from typing import Dict

import torch
from monai.data import ThreadDataLoader, partition_dataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset
from monai.transforms import (  # RandShiftIntensityd,; Resized,
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    AsChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from sw_interactive_segmentation.utils.transforms import (
    AddGuidanceSignal,
    AddGuidance,
    CheckTheAmountOfInformationLossByCropd,
    FindDiscrepancyRegions,
    InitLoggerd,
    NoOpd,
    NormalizeLabelsInDatasetd,
    PrintGPUUsaged,
    SplitPredsLabeld,
    threshold_foreground,
    AddEmptySignalChannels,
)

logger = logging.getLogger("sw_interactive_segmentation")

AUTPET_SPACING = [2.03642011, 2.03642011, 3.0]
MSD_SPLEEN_SPACING = [2 * 0.79296899, 2 * 0.79296899, 5.0]


def get_pre_transforms(labels: Dict, device, args, input_keys=("image", "label")):
    spacing = AUTPET_SPACING if args.dataset == "AutoPET" else MSD_SPLEEN_SPACING
    cpu_device = torch.device("cpu")
    
    # Input keys have to be ["image", "label"] for train, and least ["image"] for val
    if args.dataset == "AutoPET":
        t_train = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                args
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            AsChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(
                keys="label", label_names=labels, device=cpu_device
            ),
            # PrintDatad(),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=spacing),
            CropForegroundd(
                keys=input_keys,
                source_key="image",
                select_fn=threshold_foreground,
            ),
            ScaleIntensityRanged(
                keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True
            ),  # 0.05 and 99.95 percentiles of the spleen HUs
            # Random Transforms #
            # allow_smaller=True not necessary for the default AUTOPET split, just there for safety so that training does not get interrupted
            RandCropByPosNegLabeld(
                keys=input_keys,
                label_key="label",
                spatial_size=args.train_crop_size,
                pos=0.6,
                neg=0.4,
                allow_smaller=True,
            )
            if args.train_crop_size is not None
            else NoOpd(),
            DivisiblePadd(keys=input_keys, k=64, value=0)
            if args.inferer == "SimpleInferer"
            else NoOpd(),  # UNet needs this
            RandFlipd(keys=input_keys, spatial_axis=[0], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[1], prob=0.10),
            RandFlipd(keys=input_keys, spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=input_keys, prob=0.10, max_k=3),
            AddEmptySignalChannels(keys=input_keys, device=device, label_names=labels),
            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
            # ClearGPUMemoryd(device=device, garbage_collection=True) if args.gpu_size == "small" else ClearGPUMemoryd(device=device),
        ]
        t_val = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(
                args
            ),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            AsChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(
                keys="label", label_names=labels, device=cpu_device
            ) if "label" in input_keys else NoOpd(),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(
                keys=input_keys, pixdim=spacing
            ),  # 2-factor because of the spatial size
            CheckTheAmountOfInformationLossByCropd(
                keys="label", roi_size=args.val_crop_size, label_names=labels
            ) if "label" in input_keys else NoOpd(),
            CropForegroundd(
                keys=input_keys,
                source_key="image",
                select_fn=threshold_foreground,
            ),
            CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            if args.val_crop_size is not None
            else NoOpd(),
            ScaleIntensityRanged(
                keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True
            ),  # 0.05 and 99.95 percentiles of the spleen HUs
            DivisiblePadd(keys=input_keys, k=64, value=0)
            if args.inferer == "SimpleInferer"
            else NoOpd(),
            AddEmptySignalChannels(keys=input_keys, device=device, label_names=labels),
            # EnsureTyped(keys=("image", "label"), device=cpu_device, track_meta=False),
            # PrintGPUUsaged(device=device, name="pre"),
        ]
    # TODO fix and reenable the part below
    # else:  # MSD Spleen
    #     t_train = [
    #         LoadImaged(keys=("image", "label"), reader="ITKReader"),
    #         ToTensord(keys=("image", "label"), device=device),
    #         EnsureChannelFirstd(keys=("image", "label")),
    #         NormalizeLabelsInDatasetd(keys="label", label_names=labels, device=device),
    #         Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         Spacingd(keys=["image", "label"], pixdim=spacing),
    #         ScaleIntensityRanged(
    #             keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True
    #         ),  # 0.05 and 99.95 percentiles of the spleen HUs
    #         DivisiblePadd(keys=["image", "label"], k=64, value=0),  # Needed for DynUNet
    #         # Random Transforms #
    #         RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
    #         RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
    #         RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
    #         RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
    #         RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
    #         Resized(
    #             keys=("image", "label"),
    #             spatial_size=[128, 128, -1],
    #             mode=("area", "nearest"),
    #         ),  # downsampled from 512x512x-1 to fit into memory
    #         # Transforms for click simulation
    #         FindAllValidSlicesMissingLabelsd(keys="label", sids="sids", device=device),
    #         AddInitialSeedPointMissingLabelsd(
    #             keys="label", guidance_key="guidance", sids_key="sids", device=device
    #         ),
    #         # ToTensord(keys=("image", "guidance"), device=device),
    #         ToTensord(keys=("image"), device=device),
    #         AddGuidanceSignalDeepEditd(
    #             keys="image",
    #             guidance_key="guidance",
    #             sigma=args.sigma,
    #             disks=args.disks,
    #             edt=args.edt,
    #             gdt=args.gdt,
    #             gdt_th=args.gdt_th,
    #             exp_geos=args.exp_geos,
    #             adaptive_sigma=args.adaptive_sigma,
    #             device=device,
    #             spacing=spacing,
    #         ),
    #         # ToTensord(keys=("image", "label"), device=torch.device('cpu')), # TODO: check why we need this on the CPU
    #     ]
    #     t_val = [
    #         LoadImaged(keys=("image", "label"), reader="ITKReader"),
    #         ToTensord(keys=("image", "label"), device=device),
    #         EnsureChannelFirstd(keys=("image", "label")),
    #         NormalizeLabelsInDatasetd(keys="label", label_names=labels, device=device),
    #         Orientationd(keys=["image", "label"], axcodes="RAS"),
    #         Spacingd(keys=["image", "label"], pixdim=spacing),
    #         ScaleIntensityRanged(
    #             keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True
    #         ),  # 0.05 and 99.95 percentiles of the spleen HUs
    #         DivisiblePadd(keys=["image", "label"], k=64, value=0),  # Needed for DynUNet
    #         Resized(
    #             keys=("image", "label"),
    #             spatial_size=[256, 256, -1],
    #             mode=("area", "nearest"),
    #         ),  # downsampled from 512x512x-1 to fit into memory
    #         # Transforms for click simulation
    #         # ToTensord(keys=("image", "guidance", "label"), device=device),
    #         FindAllValidSlicesMissingLabelsd(keys="label", sids="sids", device=device),
    #         AddInitialSeedPointMissingLabelsd(
    #             keys="label", guidance_key="guidance", sids_key="sids", device=device
    #         ),
    #         AddGuidanceSignalDeepEditd(
    #             keys="image",
    #             guidance_key="guidance",
    #             sigma=args.sigma,
    #             disks=args.disks,
    #             edt=args.edt,
    #             gdt=args.gdt,
    #             gdt_th=args.gdt_th,
    #             exp_geos=args.exp_geos,
    #             adaptive_sigma=args.adaptive_sigma,
    #             device=device,
    #             spacing=spacing,
    #         ),
    #         # ToTensord(keys=("image", "label"), device=torch.device('cpu')),
    # ]
    return Compose(t_train), Compose(t_val)


def get_click_transforms(device, args):
    spacing = AUTPET_SPACING if args.dataset == "AutoPET" else MSD_SPLEEN_SPACING
    t = [
        InitLoggerd(args),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        FindDiscrepancyRegions(
            keys="label", pred_key="pred", discrepancy_key="discrepancy", device=device
        ),
        AddGuidance(
            keys="NA",
            guidance_key="guidance",
            discrepancy_key="discrepancy",
            probability_key="probability",
            device=device,
        ),
        AddGuidanceSignal(
            keys="image",
            guidance_key="guidance",
            sigma=args.sigma,
            disks=args.disks,
            edt=args.edt,
            gdt=args.gdt,
            gdt_th=args.gdt_th,
            exp_geos=args.exp_geos,
            adaptive_sigma=args.adaptive_sigma,
            device=device,
            spacing=spacing,
        ),  # Overwrites the image entry
        # ClearGPUMemoryd(device=device) if args.gpu_size == "small" else NoOpd(),
        # PrintGPUUsaged(device=device, name="click"),
    ]

    return Compose(t)


def get_post_transforms(labels, device):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
        # PrintGPUUsaged(device=device, name="post"),
    ]
    return Compose(t)

def get_val_post_transforms(labels, device):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
        # PrintGPUUsaged(device=device, name="post"),
    ]
    return Compose(t)



def get_loaders(args, pre_transforms_train, pre_transforms_val):
    all_images = sorted(glob.glob(os.path.join(args.input, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input, "labelsTr", "*.nii.gz")))

    if args.dataset == "AutoPET":
        test_images = sorted(
            glob.glob(os.path.join(args.input, "imagesTs", "*.nii.gz"))
        )
        test_labels = sorted(
            glob.glob(os.path.join(args.input, "labelsTs", "*.nii.gz"))
        )

        # TODO try if this impacts training?!?
        # with open('utils/zero_autopet.txt', 'r') as f:
        #    bad_images = [el.rstrip() for el in f.readlines()] # Filter out crops without any labels

        datalist = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(all_images, all_labels)
        ]  # if image_name not in bad_images]
        val_datalist = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ]  # if image_name not in bad_images]
        train_datalist = datalist
        # For debugging with small dataset size
        train_datalist = (
            train_datalist[0 : args.limit] if args.limit else train_datalist
        )
        val_datalist = val_datalist[0 : args.limit] if args.limit else val_datalist

    else:  # MSD_Spleen
        datalist = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(all_images, all_labels)
        ]
        # For debugging with small dataset size
        datalist = datalist[0 : args.limit] if args.limit else datalist
        train_datalist, val_datalist = partition_dataset(
            datalist,
            ratios=[args.split, (1 - args.split)],
            shuffle=True,
            seed=args.seed,
        )

    total_l = len(train_datalist) + len(val_datalist)

    train_ds = PersistentDataset(
        train_datalist, pre_transforms_train, cache_dir=args.cache_dir
    )
    # Need persistens workers to fix Cuda worker error: "[W CUDAGuardImpl.h:46] Warning: CUDA warning: driver shutting down (function uncheckedGetDevice"
    train_loader = ThreadDataLoader(
        train_ds,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=1,
        multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info(
        "{} :: Total Records used for Training is: {}/{}".format(
            args.gpu, len(train_ds), total_l
        )
    )

    val_ds = PersistentDataset(
        val_datalist, pre_transforms_val, cache_dir=args.cache_dir
    )

    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
        multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info(
        "{} :: Total Records used for Validation is: {}/{}".format(
            args.gpu, len(val_ds), total_l
        )
    )

    return train_loader, val_loader
