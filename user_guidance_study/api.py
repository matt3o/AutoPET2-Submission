import logging

from monai.networks.nets.dynunet import DynUNet
from monai.data import set_track_meta

logger = logging.getLogger("interactive_segmentation")

from utils.utils import (
    get_click_transforms,
    get_loaders,
    get_post_transforms,
    get_pre_transforms,
)

from utils.helper import count_parameters

__all__ = [
    'get_optimizer',
    'get_click_transforms',
    'get_click_transforms',
    'get_post_transforms',
    'get_loaders',
    'get_loss_function',
    'get_network',
    'get_inferers'
]


def get_optimizer(optimizer: str, lr: float):
    # OPTIMIZER
    if optimizer == "Novograd":
        optimizer = Novograd(network.parameters(), lr)
    elif optimizer == "Adam":  # default
        optimizer = torch.optim.Adam(network.parameters(), lr)
    return optimizer

def get_loss_function():
    # squared_pred enables much faster convergence, possibly even better results in the long run
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
    return loss_function


def get_network(network_str: str = "dynunet", labels: Iterable):
    if network_str == "dynunet":
        network = DynUNet(
            spatial_dims=3,
            # 1 dim for the image, the other ones for the signal per label with is the size of image
            in_channels=1 + len(labels),
            out_channels=len(labels),
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    elif network_str == "smalldynunet":
        network = DynUNet(
            spatial_dims=3,
            # 1 dim for the image, the other ones for the signal per label with is the size of image
            in_channels=1 + len(labels),
            out_channels=len(labels),
            kernel_size=[3, 3, 3],
            strides=[1, 2, [2, 2, 1]],
            upsample_kernel_size=[2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )
    set_track_meta(False)
    logger.info(f"Selected network {network_str}")
    logger.info(f"Number of parameters: {count_parameters(network):,}")

    return network


def get_inferers(inferer: str = "SlidingWindowInferer", 
                sw_roi_size, 
                train_crop_size, 
                val_crop_size, 
                train_sw_batch_size, 
                val_sw_batch_size):
    if inferer == "SimpleInferer":
        train_inferer = SimpleInferer()
        eval_inferer = SimpleInferer()
    elif inferer == "SlidingWindowInferer":
        # train_batch_size is limited due to this bug: https://github.com/Project-MONAI/MONAI/issues/6628
        assert train_crop_size is not None
        train_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [
                        round(train_crop_size[i] / sw_roi_size[i])
                        for i in range(len(sw_roi_size))
                    ],
                ),
                train_sw_batch_size,
            ),
        )
        logger.info(f"{train_batch_size=}")
        average_sample_shape = (300, 300, 400)
        if val_crop_size != "None":
            average_sample_shape = val_crop_size
            
        val_batch_size = max(
            1,
            min(
                reduce(
                    lambda x, y: x * y,
                    [
                        round(average_sample_shape[i] / sw_roi_size[i])
                        for i in range(len(sw_roi_size))
                    ],
                ),
                val_sw_batch_size,
            ),
        )
        logger.info(f"{val_batch_size=}")

        
        train_inferer = SlidingWindowInferer(
            roi_size=sw_roi_size,
            sw_batch_size=train_batch_size,
            mode="gaussian",
            cache_roi_weight_map=True,
        )
        eval_inferer = SlidingWindowInferer(
            roi_size=sw_roi_size,
            sw_batch_size=val_batch_size,
            mode="gaussian",
            cache_roi_weight_map=True,
        )
    return train_inferer, eval_inferer

def get_scheduler(optimizer, scheduler_str: str = "MultiStepLR", epochs_to_run):
    if scheduler_str == "MultiStepLR":
        steps = 4
        steps_per_epoch = round(epochs_to_run / steps)
        if steps_per_epoch < 1:
            logger.error(f"Chosen number of epochs {epochs_to_run}/{steps} < 0")
            milestones = range(0, epochs_to_run)
        else:
            milestones = [
                num
                for num in range(0, epochs_to_run)
                if num % round(steps_per_epoch) == 0
            ][1:]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.333
        )
    elif scheduler_str == "PolynomialLR":
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=epochs_to_run, power=2
        )
    elif scheduler_str == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_to_run, eta_min=1e-6
        )
    return lr_scheduler
