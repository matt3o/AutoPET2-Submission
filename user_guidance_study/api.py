from __future__ import annotations

import logging
from collections import OrderedDict
from functools import reduce
from typing import Iterable, List

import cupy as cp
import torch
from ignite.engine import Events
from ignite.handlers import TerminateOnNan
from monai.data import set_track_meta
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointLoader,
    CheckpointSaver,
    GarbageCollector,
    IgniteMetric,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    ValidationHandler,
    from_engine,
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import LossMetric
from monai.networks.nets.dynunet import DynUNet
from monai.optimizers.novograd import Novograd
from monai.utils import set_determinism

from utils.helper import count_parameters, run_once
from utils.interaction import Interaction
from utils.utils import get_click_transforms, get_loaders, get_post_transforms, get_pre_transforms

logger = logging.getLogger("sw_interactive_segmentation")
output_dir = None

__all__ = [
    "get_optimizer",
    "get_click_transforms",
    "get_click_transforms",
    "get_post_transforms",
    "get_loaders",
    "get_loss_function",
    "get_network",
    "get_inferers",
    "get_scheduler",
    "get_train_handlers",
    "get_val_handlers",
    "get_key_val_metrics",
    "get_key_train_metrics",
    "get_trainer",
    "get_evaluator",
]


def get_optimizer(optimizer: str, lr: float, network):
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


def get_network(network_str: str, labels: Iterable):
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
    logger.info(f"Selected network {network_str.__class__.__qualname__}")
    logger.info(f"Number of parameters: {count_parameters(network):,}")

    return network


def get_inferers(
    inferer: str,
    sw_roi_size,
    train_crop_size,
    val_crop_size,
    train_sw_batch_size,
    val_sw_batch_size,
):
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
        if val_crop_size is not None:
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


def get_scheduler(optimizer, scheduler_str: str, epochs_to_run: int):
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


def get_val_handlers(sw_roi_size: List, inferer: str, gpu_size: str):
    if sw_roi_size[0] < 128:
        val_trigger_event = (
            Events.ITERATION_COMPLETED(every=2)
            if gpu_size == "large"
            else Events.ITERATION_COMPLETED(every=1)
        )
    else:
        val_trigger_event = (
            Events.ITERATION_COMPLETED(every=2)
            if gpu_size == "large"
            else Events.ITERATION_COMPLETED(every=1)
        )

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if inferer == "SlidingWindowInferer":
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=val_trigger_event)
        val_handlers.append(iteration_gc)

    return val_handlers


def get_train_handlers(
    lr_scheduler,
    evaluator,
    val_freq,
    eval_only: bool,
    sw_roi_size: List,
    inferer: str,
    gpu_size: str,
):
    if sw_roi_size[0] < 128:
        train_trigger_event = (
            Events.ITERATION_COMPLETED(every=10)
            if gpu_size == "large"
            else Events.ITERATION_COMPLETED(every=1)
        )
    else:
        train_trigger_event = (
            Events.ITERATION_COMPLETED(every=10)
            if gpu_size == "large"
            else Events.ITERATION_COMPLETED(every=5)
        )

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator,
            interval=val_freq,
            epoch_level=(not eval_only),
        ),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
        # End of epoch GarbageCollection
        GarbageCollector(log_level=10),
    ]
    if inferer == "SlidingWindowInferer":
        # https://github.com/Project-MONAI/MONAI/issues/3423
        iteration_gc = GarbageCollector(log_level=10, trigger_event=train_trigger_event)
        train_handlers.append(iteration_gc)

    return train_handlers


def get_key_val_metrics():
    loss_function_metric = DiceCELoss(softmax=True, squared_pred=True)
    metric_fn = LossMetric(
        loss_fn=loss_function_metric, reduction="mean", get_not_nans=False
    )
    ignite_metric = IgniteMetric(
        metric_fn=metric_fn,
        output_transform=from_engine(["pred", "label"]),
        save_details=False,
    )

    all_val_metrics = OrderedDict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    all_val_metrics["val_mean_dice_ce_loss"] = ignite_metric

    # Disabled since it led to weird artefacts in the Tensorboard diagram
    # for key_label in args.labels:
    #     if key_label != "background":
    #         all_val_metrics[key_label + "_dice"] = MeanDice(
    #             output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
    #         )

    return all_val_metrics


def get_key_train_metrics():
    all_train_metrics = get_key_val_metrics()
    all_train_metrics["train_dice"] = all_train_metrics.pop("val_mean_dice")
    all_train_metrics["train_dice_ce_loss"] = all_train_metrics.pop(
        "val_mean_dice_ce_loss"
    )
    return all_train_metrics


def get_evaluator(
    args,
    network,
    inferer,
    device,
    val_loader,
    loss_function,
    click_transforms,
    post_transform,
    key_val_metric,
) -> SupervisedEvaluator:
    init(args)

    # if network is None:
    #     network = get_network(args.network, args.labels).to(device)
    # if inferer is None:
    #     inferer = get_inferer(args.inferer)

    # val_handlers = get_val_handlers(sw_roi_size=args.sw_roi_size, inferer=args.inferer, gpu_size=args.gpu_size)

    # if loss_function is None:
    #     loss_function = get_loss_function()

    # if val_loader is None:
    #     pre_transforms_train, pre_transforms_val = get_pre_transforms(
    #     args.labels, device, args
    #     )
    #     _, val_loader = get_loaders(
    #     args, pre_transforms_train, pre_transforms_val
    #     )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            args=args,
            loss_function=loss_function,
            post_transform=post_transform,
            click_generation_strategy=args.val_click_generation,
            stopping_criterion=args.val_click_generation_stopping_criterion,
        ),
        inferer=inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=key_val_metric,
        val_handlers=get_val_handlers(
            sw_roi_size=args.sw_roi_size, inferer=args.inferer, gpu_size=args.gpu_size
        ),
    )
    return evaluator


def get_trainer(args) -> List[SupervisedTrainer, SupervisedEvaluator, List]:
    init(args)
    device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_train, pre_transforms_val = get_pre_transforms(
        args.labels, device, args
    )
    click_transforms = get_click_transforms(device, args)
    post_transform = get_post_transforms(args.labels, device)
    train_loader, val_loader = get_loaders(
        args, pre_transforms_train, pre_transforms_val
    )

    network = get_network(args.network, args.labels).to(device)
    train_inferer, eval_inferer = get_inferers(
        args.inferer,
        args.sw_roi_size,
        args.train_crop_size,
        args.val_crop_size,
        args.train_sw_batch_size,
        args.val_sw_batch_size,
    )
    loss_function = get_loss_function()
    optimizer = get_optimizer(args.optimizer, args.learning_rate, network)
    lr_scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)
    train_metrics = get_key_train_metrics()
    val_metrics = get_key_val_metrics()

    evaluator = get_evaluator(
        args,
        network=network,
        inferer=eval_inferer,
        device=device,
        val_loader=val_loader,
        loss_function=loss_function,
        click_transforms=click_transforms,
        post_transform=post_transform,
        key_val_metric=val_metrics,
    )

    train_handlers = get_train_handlers(
        lr_scheduler,
        evaluator,
        args.val_freq,
        args.eval_only,
        args.sw_roi_size,
        args.inferer,
        args.gpu_size,
    )
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            train=True,
            label_names=args.labels,
            max_interactions=args.max_train_interactions,
            args=args,
            loss_function=loss_function,
            post_transform=post_transform,
            click_generation_strategy=args.train_click_generation,
            stopping_criterion=args.train_click_generation_stopping_criterion,
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=train_metrics,
        train_handlers=train_handlers,
    )

    save_dict = get_save_dict(trainer, network, optimizer, lr_scheduler)
    CheckpointSaver(
        save_dir=args.output,
        save_dict=save_dict,
        save_interval=args.save_interval,
        save_final=True,
        final_filename="checkpoint.pt",
        n_saved=2,
    ).attach(trainer)
    CheckpointSaver(
        save_dir=args.output,
        save_dict=save_dict,
        save_key_metric=True,
        save_final=True,
        save_interval=args.save_interval,
        final_filename="pretrained_deepedit_" + args.network + ".pt",
    ).attach(evaluator)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    if args.resume_from != "None":
        logger.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        checkpoint = torch.load(args.resume_from)

        for key in save_dict:
            # If it fails: the file may be broken or incompatible (e.g. evaluator has not been run)
            assert (
                key in checkpoint
            ), f"key {key} has not been found in the save_dict! \n file keys: {checkpoint.keys}"

        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        handler = CheckpointLoader(
            load_path=args.resume_from, load_dict=save_dict, map_location=map_location
        )
        handler(trainer)

    return trainer, evaluator, train_metrics, val_metrics


def get_save_dict(trainer, network, optimizer, lr_scheduler):
    save_dict = {
        "trainer": trainer,
        "net": network,
        "opt": optimizer,
        "lr": lr_scheduler,
    }
    return save_dict


@run_once
def init(args):
    global output_dir
    # for OOM debugging
    output_dir = args.output

    set_determinism(seed=args.seed)
    with cp.cuda.Device(args.gpu):
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=14 * 1024**3)
        cp.random.seed(seed=args.seed)


def oom_observer(device, alloc, device_alloc, device_free):
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    print("Tips: \nReduce sw_batch_size if there is an OOM (maybe even roi_size)")
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f"{output_dir}/oom_snapshot.pickle", "wb"))
    # logger.critical(snapshot)
    torch.cuda.memory._save_memory_usage(
        filename=f"{output_dir}/memory.svg", snapshot=snapshot
    )
    torch.cuda.memory._save_segment_usage(
        filename=f"{output_dir}/segments.svg", snapshot=snapshot
    )
