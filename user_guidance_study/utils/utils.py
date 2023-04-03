from utils.transforms import (
    AddGuidanceSignalDeepEditd,
    AddRandomGuidanceDeepEditd,
    FindDiscrepancyRegionsDeepEditd,
    NormalizeLabelsInDatasetd,
    FindAllValidSlicesMissingLabelsd,
    AddInitialSeedPointMissingLabelsd,
    SplitPredsLabeld,
)

from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    DivisiblePadd,
    ToNumpyd,
    ToTensord,
    CenterSpatialCropd
)
from monai.data import partition_dataset
from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset

import torch
import glob
import os
import logging

def get_pre_transforms(labels, device, args):
    spacing = [2.03642011, 2.03642011, 3.        ] if args.dataset == 'AutoPET' else [2 * 0.79296899, 2 * 0.79296899, 5.        ]
    if args.dataset == 'AutoPET':
        t_train = [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=labels),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 256)),
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
            ### Random Transforms ###
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet

            # Transforms for click simulation
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            ToTensord(keys=("image", "guidance"), device=device),
            AddGuidanceSignalDeepEditd(keys="image",
                                        guidance="guidance",
                                        sigma=args.sigma,
                                        disks=args.disks,
                                        edt=args.edt,
                                        gdt=args.gdt,
                                        gdt_th=args.gdt_th,
                                        exp_geos=args.exp_geos,
                                        adaptive_sigma=args.adaptive_sigma,
                                        device=device, spacing=spacing),        #
            ToTensord(keys=("image", "label"), device=torch.device('cpu')),
        ]
        t_val = [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=labels),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing), # 2-factor because of the spatial size
            CenterSpatialCropd(keys=["image", "label"], roi_size=(192, 192, 256)),
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
            DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet
            # Transforms for click simulation
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            ToTensord(keys=("image", "guidance"), device=device),
            AddGuidanceSignalDeepEditd(keys="image",
                                        guidance="guidance",
                                        sigma=args.sigma,
                                        disks=args.disks,
                                        edt=args.edt,
                                        gdt=args.gdt,
                                        gdt_th=args.gdt_th,
                                        exp_geos=args.exp_geos,
                                        adaptive_sigma=args.adaptive_sigma,
                                        device=device, spacing=spacing),
            ToTensord(keys=("image", "label"), device=torch.device('cpu')),
        ]
    else: # MSD Spleen
        t_train = [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=labels),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing),
            ScaleIntensityRanged(keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
            DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet
            ### Random Transforms ###
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.10),
            RandFlipd(keys=("image", "label"), spatial_axis=[2], prob=0.10),
            RandRotate90d(keys=("image", "label"), prob=0.10, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.10, prob=0.50),
            Resized(keys=("image", "label"), spatial_size=[256, 256, -1], mode=("area", "nearest")), # downsampled from 512x512x-1 to fit into memory
            # Transforms for click simulation
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            ToTensord(keys=("image", "guidance"), device=device),
            AddGuidanceSignalDeepEditd(keys="image",
                                        guidance="guidance",
                                        sigma=args.sigma,
                                        disks=args.disks,
                                        edt=args.edt,
                                        gdt=args.gdt,
                                        gdt_th=args.gdt_th,
                                        exp_geos=args.exp_geos,
                                        adaptive_sigma=args.adaptive_sigma,
                                        device=device, spacing=spacing),
            ToTensord(keys=("image", "label"), device=torch.device('cpu')), # TODO: check why we need this on the CPU
        ]
        t_val = [
            LoadImaged(keys=("image", "label"), reader="ITKReader"),
            EnsureChannelFirstd(keys=("image", "label")),
            NormalizeLabelsInDatasetd(keys="label", label_names=labels),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing),
            ScaleIntensityRanged(keys="image", a_min=-224, a_max=212, b_min=0.0, b_max=1.0, clip=True), # 0.05 and 99.95 percentiles of the spleen HUs
            DivisiblePadd(keys=["image", "label"], k=64, value=0), # Needed for DynUNet

            Resized(keys=("image", "label"), spatial_size=[256, 256, -1], mode=("area", "nearest")), # downsampled from 512x512x-1 to fit into memory
            # Transforms for click simulation
            FindAllValidSlicesMissingLabelsd(keys="label", sids="sids"),
            AddInitialSeedPointMissingLabelsd(keys="label", guidance="guidance", sids="sids"),
            ToTensord(keys=("image", "guidance"), device=device),
            AddGuidanceSignalDeepEditd(keys="image",
                                        guidance="guidance",
                                        sigma=args.sigma,
                                        disks=args.disks,
                                        edt=args.edt,
                                        gdt=args.gdt,
                                        gdt_th=args.gdt_th,
                                        exp_geos=args.exp_geos,
                                        adaptive_sigma=args.adaptive_sigma,
                                        device=device, spacing=spacing),
            ToTensord(keys=("image", "label"), device=torch.device('cpu')),
        ]
    return Compose(t_train), Compose(t_val)

def get_click_transforms(device, args):
    spacing = [2.03642011, 2.03642011, 3.        ] if args.dataset == 'AutoPET' else [2 * 0.79296899, 2 * 0.79296899, 5.        ] # 2-factor because of the spatial size

    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        ToNumpyd(keys=("image", "label", "pred")),
        # Transforms for click simulation
        FindDiscrepancyRegionsDeepEditd(keys="label", pred="pred", discrepancy="discrepancy"),
        AddRandomGuidanceDeepEditd(
            keys="NA",
            guidance="guidance",
            discrepancy="discrepancy",
            probability="probability",
        ),
        ToTensord(keys=("image", "guidance"), device=device),
        AddGuidanceSignalDeepEditd(keys="image",
                                    guidance="guidance",
                                    sigma=args.sigma,
                                    disks=args.disks,
                                    edt=args.edt,
                                    gdt=args.gdt,
                                    gdt_th=args.gdt_th,
                                    exp_geos=args.exp_geos,
                                    adaptive_sigma=args.adaptive_sigma,
                                    device=device, spacing=spacing),        #
        ToTensord(keys=("image", "label"), device=torch.device('cpu')),
    ]

    return Compose(t)

def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)

def get_loaders(args, pre_transforms_train, pre_transforms_val):
    all_images = sorted(glob.glob(os.path.join(args.input, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input, "labelsTr", "*.nii.gz")))

    if args.dataset == 'AutoPET':
        test_images = sorted(glob.glob(os.path.join(args.input, "imagesTs", "*.nii.gz")))
        test_labels = sorted(glob.glob(os.path.join(args.input, "labelsTs", "*.nii.gz")))

        with open('utils/zero_autopet.txt', 'r') as f:
            bad_images = [el.rstrip() for el in f.readlines()] # Filter out crops without any labels

        datalist = [{"image": image_name, "label": label_name} for image_name, label_name in
                    zip(all_images, all_labels) if image_name not in bad_images]
        val_datalist = [{"image": image_name, "label": label_name} for image_name, label_name in
                    zip(test_images, test_labels) if image_name not in bad_images]
        train_datalist = datalist
        # For debugging with small dataset size
        train_datalist = train_datalist[0: args.limit] if args.limit else train_datalist
        val_datalist = val_datalist[0: args.limit] if args.limit else val_datalist

    else: # MSD_Spleen
        datalist = [{"image": image_name, "label": label_name} for image_name, label_name in
                    zip(all_images, all_labels)]
        # For debugging with small dataset size
        datalist = datalist[0: args.limit] if args.limit else datalist
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
    train_loader = DataLoader(
        train_ds, shuffle=True, num_workers=args.num_workers, batch_size=1, multiprocessing_context='spawn',
    )
    logging.info(
        "{} :: Total Records used for Training is: {}/{}".format(
            args.gpu, len(train_ds), total_l
        )
    )

    val_ds = PersistentDataset(val_datalist, pre_transforms_val, cache_dir=args.cache_dir)

    val_loader = DataLoader(val_ds, num_workers=args.num_workers, batch_size=1, multiprocessing_context='spawn')
    logging.info(
        "{} :: Total Records used for Validation is: {}/{}".format(
            args.gpu, len(val_ds), total_l
        )
    )

    return train_loader, val_loader
