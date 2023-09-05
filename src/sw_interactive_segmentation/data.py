from __future__ import annotations

import glob
import logging
import os
from typing import Dict
# from collections import OrderedDict

import torch
from monai.data import ThreadDataLoader, partition_dataset

# from monai.data.dataloader import DataLoader
from monai.data.dataset import PersistentDataset  # , Dataset
from monai.transforms import (  # RandShiftIntensityd,; Resized,; ScaleIntensityRanged,
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    ToDeviced,
    ToTensord,
    MeanEnsembled,
)
from monai.utils.enums import CommonKeys
from monai.apps import CrossValidation

from sw_interactive_segmentation.transforms import (  # PrintGPUUsaged,; PrintDatad,
    AddEmptySignalChannels,
    AddGuidance,
    AddGuidanceSignal,
    CheckTheAmountOfInformationLossByCropd,
    FindDiscrepancyRegions,
    InitLoggerd,
    NoOpd,
    NormalizeLabelsInDatasetd,
    SplitPredsLabeld,
    threshold_foreground,
)

logger = logging.getLogger("sw_interactive_segmentation")

AUTOPET_SPACING = [2.03642011, 2.03642011, 3.0]
MSD_SPLEEN_SPACING = [2 * 0.79296899, 2 * 0.79296899, 5.0]
HECKTOR_SPACING = [4, 4, 4]


def get_pre_transforms(labels: Dict, device, args, input_keys=("image", "label")):
    return Compose(get_pre_transforms_train_as_list(labels, device, args, input_keys)), Compose(
        get_pre_transforms_val_as_list(labels, device, args, input_keys)
    )


def get_pre_transforms_train_as_list(labels: Dict, device, args, input_keys=("image", "label")):
    cpu_device = torch.device("cpu")
    if args.dataset == "AutoPET" or args.dataset == "AutoPET2":
        spacing = AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        spacing = HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        spacing = MSD_SPLEEN_SPACING

    PET_dataset_names = ["AutoPET", "AutoPET2", "AutoPET_merged", "HECKTOR"]
    # data Input keys have to be ["image", "label"] for train, and least ["image"] for val
    if args.dataset in PET_dataset_names:
        t_train = [
            # Initial transforms on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(args),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(
                keys=input_keys,
                reader="ITKReader",
                image_only=False,
                simple_keys=True,
            ),
            ToTensord(keys=input_keys, device=cpu_device, track_meta=True),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=spacing),
            CropForegroundd(
                keys=input_keys,
                source_key="image",
                select_fn=threshold_foreground,
            )
            if not args.dont_crop_foreground
            else NoOpd(),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True)
            if not args.use_scale_intensity_range_percentiled
            else ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            # Random Transforms #
            # allow_smaller=True not necessary for the default AUTOPET split of (224,)**3, just there for safety so that training does not get interrupted
            RandCropByPosNegLabeld(
                keys=input_keys,
                label_key="label",
                spatial_size=args.train_crop_size,
                pos=args.positive_crop_rate,
                neg=1 - args.positive_crop_rate,
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
            AddEmptySignalChannels(keys=input_keys, device=cpu_device) if not args.non_interactive else NoOpd(),
            # Move to GPU
            # WARNING: Activating the line below leads to minimal gains in performance
            # However you are buying these gains with a lot of weird errors and problems
            # So my recommendation after months of fiddling is to leave this off
            # Until MONAI has fixed the underlying issues
            # ToTensord(keys=("image", "label"), device=device, track_meta=False),
        ]
    return t_train


def get_pre_transforms_val_as_list(labels: Dict, device, args, input_keys=("image", "label")):
    cpu_device = torch.device("cpu")
    if args.dataset == "AutoPET" or args.dataset == "AutoPET2":
        spacing = AUTOPET_SPACING
    elif args.dataset == "HECKTOR":
        spacing = HECKTOR_SPACING
    elif args.dataset == "MSD_Spleen":
        spacing = MSD_SPLEEN_SPACING

    PET_dataset_names = ["AutoPET", "AutoPET2", "AutoPET_merged", "HECKTOR"]
    # data Input keys have to be ["image", "label"] for train, and least ["image"] for val
    if args.dataset in PET_dataset_names:
        t_val = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(args),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=spacing),  # 2-factor because of the spatial size
            CheckTheAmountOfInformationLossByCropd(
                keys="label", roi_size=args.val_crop_size, crop_foreground=(not args.dont_crop_foreground)
            )
            if "label" in input_keys
            else NoOpd(),
            # CropForegroundd(
            #     keys=input_keys,
            #     source_key="image",
            #     select_fn=threshold_foreground,
            # ) if not args.dont_crop_foreground else NoOpd(),
            # CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            # if args.val_crop_size is not None
            # else NoOpd(),
            # 0.05 and 99.95 percentiles of the spleen HUs, either manually or automatically
            ScaleIntensityRanged(keys="image", a_min=0, a_max=43, b_min=0.0, b_max=1.0, clip=True)
            if not args.use_scale_intensity_range_percentiled
            else ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            DivisiblePadd(keys=input_keys, k=64, value=0) if args.inferer == "SimpleInferer" else NoOpd(),
            AddEmptySignalChannels(keys=input_keys, device=cpu_device) if not args.non_interactive else NoOpd(),
            # EnsureTyped(keys=("image", "label"), device=cpu_device, track_meta=False),
        ]
    return t_val


def get_pre_transforms_val_as_list_monailabel(labels: Dict, device, args, input_keys=("image")):
    spacing = AUTOPET_SPACING if args.dataset == "AutoPET" else MSD_SPLEEN_SPACING
    cpu_device = torch.device("cpu")

    # Input keys have to be ["image", "label"] for train, and least ["image"] for val
    if args.dataset == "AutoPET":
        t_val_1 = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            InitLoggerd(args),  # necessary if the dataloader runs in an extra thread / process
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=labels, device=cpu_device),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),  # 0.05 and 99.95 percentiles of the spleen HUs
            EnsureTyped(keys=input_keys, device=device, data_type="tensor"),
        ]
        t_val_2 = [
            AddEmptySignalChannels(keys=input_keys, device=device),
            AddGuidanceSignal(
                keys=input_keys,
                sigma=1,
                disks=True,
                device=device,
            ),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=spacing),  # 2-factor because of the spatial size
            CenterSpatialCropd(keys=input_keys, roi_size=args.val_crop_size)
            if args.val_crop_size is not None
            else NoOpd(),
            DivisiblePadd(keys=input_keys, k=64, value=0) if args.inferer == "SimpleInferer" else NoOpd(),
        ]
    return t_val_1, t_val_2

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


def get_click_transforms(device, args):
    spacing = AUTOPET_SPACING if args.dataset == "AutoPET" else MSD_SPLEEN_SPACING
    t = [
        InitLoggerd(args),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),
        FindDiscrepancyRegions(keys="label", pred_key="pred", discrepancy_key="discrepancy", device=device),
        AddGuidance(
            keys="NA",
            discrepancy_key="discrepancy",
            probability_key="probability",
            device=device,
        ),
        AddGuidanceSignal(
            keys="image",
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
    ]
    return Compose(t)


def get_ensemble_transforms(labels, device, nfolds=5, weights=None):
    prediction_keys = [f"pred_{i}" for i in range(nfolds)]

    t = [
        EnsureTyped(keys=prediction_keys),
        MeanEnsembled(
            keys=prediction_keys,
            output_key="pred",
            # weights=[0.95, 0.94, 0.95, 0.94, 0.90],
        ),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
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
    ]
    return Compose(t)


def get_AutoPET_file_list(args):
    train_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTs", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTs", "*.nii.gz")))

    train_data = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
    ]
    val_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    return train_data, val_data


def get_MSD_Spleen_file_list(args):
    all_images = sorted(glob.glob(os.path.join(args.input_dir, "imagesTr", "*.nii.gz")))
    all_labels = sorted(glob.glob(os.path.join(args.input_dir, "labelsTr", "*.nii.gz")))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )
    return train_data, val_data


def get_AutoPET2_file_list(args):
    all_images = glob.glob(os.path.join(args.input, "**", "**", "SUV*.nii.gz"))
    all_labels = glob.glob(os.path.join(args.input, "**", "**", "SEG*.nii.gz"))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    return train_data, val_data


def get_HECKTOR_file_list(args):
    # Assuming this is the folder /lsdf/data/medical/HECKTOR/hecktor2022_training/
    train_images = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_training", "imagesTr", "*PT*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_training", "labelsTr", "*.nii.gz")))

    test_images = sorted(glob.glob(os.path.join(args.input_dir, "hecktor2022_testing", "imagesTs", "*.nii.gz")))

    data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

    logger.info(f"{data[-5:]=}")

    train_data, val_data = partition_dataset(
        data,
        ratios=[args.split, (1 - args.split)],
        shuffle=True,
        seed=args.seed,
    )

    test_data = [{"image": image_name} for image_name in test_images]
    return train_data, val_data, test_data


def get_data(args):
    logger.info(f"{args.dataset=}")

    test_data = []
    if args.dataset == "AutoPET":
        train_data, val_data = get_AutoPET_file_list(args)
    elif args.dataset == "MSD_Spleen":
        train_data, val_data = get_MSD_Spleen_file_list(args)
    elif args.dataset == "AutoPET2":
        train_data, val_data = get_AutoPET2_file_list(args)
    elif args.dataset == "HECKTOR":
        train_data, val_data, test_data = get_HECKTOR_file_list(args)

    # For debugging with small dataset size
    train_data = train_data[0 : args.limit] if args.limit else train_data
    val_data = val_data[0 : args.limit] if args.limit else val_data

    if args.train_on_all_samples:
        train_data += val_data
        logger.warning("All validation data has been added to the training. Validation on them no longer makes sense.")

    return train_data, val_data, test_data


def get_loaders(args, pre_transforms_train, pre_transforms_val):
    train_data, val_data, test_data = get_data(args)

    total_l = len(train_data) + len(val_data)
    train_ds = PersistentDataset(train_data, pre_transforms_train, cache_dir=args.cache_dir)
    train_loader = ThreadDataLoader(
        train_ds,
        shuffle=True,
        num_workers=args.num_workers,
        batch_size=1,
        # The two options below are needed if ToDeviced('cuda' ,..) is activated..
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Training is: {}/{}".format(args.gpu, len(train_ds), total_l))

    val_ds = PersistentDataset(val_data, pre_transforms_val, cache_dir=args.cache_dir)

    val_loader = ThreadDataLoader(
        val_ds,
        num_workers=args.num_workers,
        batch_size=1,
        # multiprocessing_context="spawn",
        # persistent_workers=True,
    )
    logger.info("{} :: Total Records used for Validation is: {}/{}".format(args.gpu, len(val_ds), total_l))

    return train_loader, val_loader


def get_cross_validation(args, nfolds, pre_transforms_train, pre_transforms_val):
    folds = list(range(nfolds))

    train_data, val_data, test_data = get_data(args)

    cvdataset = CrossValidation(
        dataset_cls=PersistentDataset,
        data=train_data,
        nfolds=nfolds,
        seed=args.seed,
        transform=pre_transforms_train,
        cache_dir=args.cache_dir,
    )

    train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
    val_dss = [cvdataset.get_dataset(folds=i, transform=pre_transforms_val) for i in range(nfolds)]

    train_loaders = [
        ThreadDataLoader(
            train_dss[i],
            shuffle=True,
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    val_loaders = [
        ThreadDataLoader(
            val_dss[i],
            num_workers=args.num_workers,
            batch_size=1,
        )
        for i in folds
    ]

    # test_ds = PersistentDataset(val_data, pre_transforms_val, cache_dir=args.cache_dir)

    # test_loader = ThreadDataLoader(
    #     val_ds,
    #     num_workers=args.num_workers,
    #     batch_size=1,
    # )

    return train_loaders, val_loaders  # , test_loader


def get_test_loader(args, file_glob="*.nii.gz"):
    labels_dir = args.labels_dir
    predictions_dir = args.predictions_dir
    predictions_glob = os.path.join(predictions_dir, file_glob)
    labels_glob = os.path.join(labels_dir, file_glob)

    test_labels = sorted(glob.glob(labels_glob))
    test_predictions = sorted(glob.glob(predictions_glob))

    test_datalist = [
        {CommonKeys.LABEL: label_name, CommonKeys.PRED: pred_name}
        for label_name, pred_name in zip(test_labels, test_predictions)
    ]
    test_datalist = test_datalist[0 : args.limit] if args.limit else test_datalist
    total_l = len(test_datalist)
    assert total_l > 0

    logger.info("{} :: Total Records used for Dataloader is: {}".format(args.gpu, total_l))

    return test_datalist


def get_test_transforms(device, labels):
    t = [
        LoadImaged(
            keys=["pred", "label"],
            reader="ITKReader",
            image_only=True,
        ),
        ToDeviced(keys=["pred", "label"], device=device),
        EnsureChannelFirstd(keys=["pred", "label"]),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(False, False),
            to_onehot=(len(labels), len(labels)),
        ),
    ]

    return Compose(t)
