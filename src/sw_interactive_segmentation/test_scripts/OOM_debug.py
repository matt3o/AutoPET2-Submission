from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import time

import monai.transforms as mt
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers.garbage_collector import GarbageCollector
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.networks.nets.dynunet import DynUNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    CuCIMd,
    DeleteItemsd,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandCuCIMd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToCupyd,
    ToNumpyd,
    ToTensord,
)

location = "/cvhci/data/AutoPET/AutoPET/"
all_images = sorted(glob.glob(os.path.join(location, "imagesTr", "*.nii.gz")))
all_labels = sorted(glob.glob(os.path.join(location, "labelsTr", "*.nii.gz")))
datalist = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)
]  # if image_name not in bad_images]

datalist = datalist[0:9]
device = "cuda"

# assert len(datalist) == 1

transform = mt.Compose(
    [
        mt.LoadImaged(keys="image", image_only=True, ensure_channel_first=True),
        mt.Resized(keys="image", spatial_size=(344, 344, 284)),
        mt.ToDeviced(keys="image", device=device),
    ]
)


def get_post_transforms(labels):
    t = [
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=(len(labels), len(labels)),
        ),
        # This transform is to check dice score per segment/label
        # SplitPredsLabeld(keys="pred"),
    ]
    return Compose(t)


train_ds = Dataset(datalist, transform)

train_loader = DataLoader(
    train_ds,
    shuffle=True,  # , num_workers=args.num_workers, batch_size=1, multiprocessing_context='spawn', persistent_workers=True,
)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
).to(device=device)

model2 = DynUNet(
    spatial_dims=3,
    # 1 dim for the image, the other ones for the signal per label with is the size of image
    in_channels=1,
    out_channels=1,
    kernel_size=[3, 3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2, [2, 2, 1]],
    upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
    norm_name="instance",
    deep_supervision=False,
    res_block=True,
    # conv1d=args.conv1d,
    # conv1s=args.conv1s,
).to(device=device)


sw_roi_size = (32, 32, 32)
sw_batch_size = 100000

chosen_model = "UNet"
if chosen_model == "UNet":
    model = model
elif chosen_model == "DynUNet":
    model = model2

batch_size_list = [1, 10, 100, 1000, 10000, 20000]
batch_size_list = [100]
eval_inferer = SlidingWindowInferer(roi_size=sw_roi_size, sw_batch_size=500, mode="gaussian", progress=True)
garbage_collector = GarbageCollector()
engine = {}
# engine["logger"] = logging.getLogger(__name__)

selected_model = model
MAX_EPOCHS = 10

optimizer = torch.optim.Adam(selected_model.parameters(), 0.001)
lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=MAX_EPOCHS, power=2)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)

trainer = SupervisedTrainer(
    device=device,
    max_epochs=MAX_EPOCHS,
    train_data_loader=train_loader,
    network=selected_model,
    optimizer=optimizer,
    loss_function=loss_function,
    inferer=eval_inferer,
    # postprocessing=post_transform,
    amp=True,
)

trainer.run()

# for item in train_loader:
#     # try:
#     #     garbage_collector(engine)
#     # except AttributeError:
#     #     pass
#     # finally:
#     #     torch.cuda.empty_cache()

#     for sw_batch_size in batch_size_list:
#         # with torch.no_grad():
#         #     with torch.cuda.amp.autocast():
#         start = time.time()
#         ret = eval_inferer(item["image"], model)
#         print(f'{chosen_model}: {sw_batch_size=}, time={(time.time()-start):.3f}')
#         ret = None
