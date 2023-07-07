# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology ###
### zdravko.marinov@kit.edu ###



import logging
import os
import sys
import time
import pathlib
from typing import Optional
import pprint
import uuid
import shutil
from pickle import dump
import signal
import math
from functools import reduce
from collections import OrderedDict

tmpdir = '/local/work/mhadlich/tmp'
if os.environ.get("SLURM_JOB_ID") is not None:
    os.environ['TMPDIR'] = tmpdir
    if not os.path.exists(tmpdir):
        pathlib.Path(tmpdir).mkdir(parents=True)

# Things needed to debug the Interaction class
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8*8192, rlimit[1]))

import pandas as pd
import torch
import cupy as cp

logger = None
output_dir = None

from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler
from utils.helper import (
    print_gpu_usage,
    get_gpu_usage,
    get_actual_cuda_index_of_device,
    get_git_information,
    gpu_usage,
    oom_observer,
    TerminationHandler,
    count_parameters,
    handle_exception,
    GPU_Thread,
)

from utils.interaction import Interaction
from utils.utils import (
    get_pre_transforms,
    get_click_transforms,
    get_post_transforms,
    get_loaders
)
from utils.transforms import (
    ClickGenerationStrategy,
    StoppingCriterion,
)

#from monai.config import print_config
#print_config()

from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    from_engine,
    GarbageCollector,
    CheckpointLoader,
    IgniteMetric,
)
from monai.metrics import LossMetric
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
# from utils.dynunet import DynUNet
from monai.networks.nets.dynunet import DynUNet

from monai.utils.profiling import ProfileHandler, WorkflowProfiler
from monai.engines.utils import IterationEvents

from ignite.engine import Engine, Events
import threading
from monai.data import set_track_meta
from monai.utils import set_determinism
from ignite.handlers import TerminateOnNan

import threading

from monai.optimizers.novograd import Novograd

from parser import parse_args
from tensorboard_logger import init_tensorboard_logger

# from monai.handlers import IgniteLossMetric

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


sys.excepthook = handle_exception

def get_network(network, labels, args):
    if network == "dynunet":
        network = DynUNet(
            spatial_dims=3,
            # 1 dim for the image, the other ones for the signal per label with is the size of image
            in_channels=1 + len(labels),
            out_channels=len(labels),
            kernel_size=[3, 3, 3, 3, 3 ,3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
            # conv1d=args.conv1d,
            # conv1s=args.conv1s,
        )
    elif network == "smalldynunet":
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
            # conv1d=args.conv1d,
            # conv1s=args.conv1s,
        )
    set_track_meta(False)
    return network


def create_trainer(args):

    set_determinism(seed=args.seed)
    with cp.cuda.Device(args.gpu):
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=14*1024**3)
        cp.random.seed(seed=args.seed)

    device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_train, pre_transforms_val = get_pre_transforms(args.labels, device, args)
    click_transforms = get_click_transforms(device, args)
    # val_click_transforms = get_click_transforms(device, args, click_generation=args.val_click_generation)
    post_transform = get_post_transforms(args.labels)

    train_loader, val_loader = get_loaders(args, pre_transforms_train, pre_transforms_val)
    numel_train = len(train_loader.dataset)
    numel_val = len(val_loader.dataset)

    # NETWORK - define training components
    network = get_network(args.network, args.labels, args).to(device)

    print('Number of parameters:', f"{count_parameters(network):,}")

     # INFERER
    if args.inferer == "SimpleInferer":
        train_inferer = SimpleInferer()
        eval_inferer = SimpleInferer()
    elif args.inferer == "SlidingWindowInferer":
        # train_batch_size is limited due to this bug: https://github.com/Project-MONAI/MONAI/issues/6628
        train_batch_size = max(1,min(reduce(lambda x, y: x*y,[round(args.train_crop_size[i] / args.sw_roi_size[i]) for i in range(len(args.sw_roi_size))]), args.train_sw_batch_size))
        logger.info(f"{train_batch_size=}")
        if args.val_crop_size != "None":
            val_batch_size = max(1,min(reduce(lambda x, y: x*y,[round((300,300,400)[i] / args.sw_roi_size[i]) for i in range(len(args.sw_roi_size))]), args.val_sw_batch_size))
            logger.info(f"{val_batch_size=}")
        # Reduce if there is an OOM
        train_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=train_batch_size, mode="gaussian", cache_roi_weight_map=True)
        eval_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=val_batch_size, mode="gaussian", cache_roi_weight_map=True)

    # OPTIMIZER
    if args.optimizer == "Novograd":
        optimizer = Novograd(network.parameters(), args.learning_rate)
    elif args.optimizer == "Adam": # default
        optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)

    MAX_EPOCHS = args.epochs
#    CURRENT_EPOCH = args.current_epoch

    # SCHEDULER
    if args.scheduler == "MultiStepLR":
        steps = 4
        steps_per_epoch = round(MAX_EPOCHS/steps)
        if steps_per_epoch < 1:
            logger.error("Chosen number of epochs {MAX_EPOCHS}/{steps} < 0")
            milestones= range(0, MAX_EPOCHS)
        else:
            milestones = [num for num in range(0, MAX_EPOCHS) if num % round(steps_per_epoch) == 0][1:]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.333)
    elif args.scheduler == "PolynomialLR":
        lr_scheduler =  torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=MAX_EPOCHS, power = 2)
    elif args.scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min = 1e-6)

    # ckpt_loader = None
    # if args.model_filepath != 'None' and not args.resume:
    #     raise UserWarning("To correctly load a network you need to add --resume otherwise no model will be loaded...")

    if args.sw_roi_size[0] < 128:
        train_trigger_event = Events.ITERATION_COMPLETED(every=10) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=1)
        val_trigger_event = Events.ITERATION_COMPLETED(every=2) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=1)
    else:
        train_trigger_event = Events.ITERATION_COMPLETED(every=10) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=5)
        val_trigger_event = Events.ITERATION_COMPLETED(every=2) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=1)
    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        # TensorBoardStatsHandler(log_dir=args.output, iteration_log=False, output_transform=lambda x: None, global_epoch_transform=lambda x: trainer.state.epoch),
        # CustomLoader(),
        # https://github.com/Project-MONAI/MONAI/issues/3423
        GarbageCollector(log_level=20, trigger_event=val_trigger_event),
    ]

    # squared_pred enables much faster convergence, possibly even better results in the long run
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
    
    loss_function_metric = DiceCELoss(softmax=True, squared_pred=True)
    metric_fn = LossMetric(loss_fn=loss_function_metric, reduction="mean", get_not_nans=False)
    ignite_metric = IgniteMetric(metric_fn=metric_fn, output_transform=from_engine(["pred", "label"]), save_details=True)

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

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_val,
            transforms=click_transforms,
            click_probability_key="probability",
            train=False,
            label_names=args.labels,
            max_interactions=args.max_val_interactions,
            args=args,
            loss_function=loss_function,
            post_transform=post_transform,
            click_generation_strategy=args.val_click_generation,
            stopping_criterion=StoppingCriterion.MAX_ITER,
        ),
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )
    # if handler is not None: 
    #     handler(evaluator)
    #     handler.attach(evaluator)


    all_train_metrics = OrderedDict()
    all_train_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]),
                                               include_background=False)
    all_train_metrics["train_dice_ce_loss"] = ignite_metric

    # if len(args.labels) > 2:
    #     for key_label in args.labels:
    #         if key_label != "background":
    #             all_train_metrics[key_label + "_dice"] = MeanDice(
    #                 output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
    #             )
    #             all_train_metrics[key_label + "_dice_with_bg"] = MeanDice(
    #                 output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=True
    #             )


    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, 
                          print_lr=True,
        ),
        ValidationHandler(
            validator=evaluator, interval=args.val_freq, epoch_level=(not args.eval_only)
        ),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
        # TensorBoardStatsHandler(
        #     log_dir=args.output,
        #     tag_name="train_loss",
        #     output_transform=from_engine(["loss"], first=True),
        # ),
        # CustomLoader(all_train_metrics),
        # https://github.com/Project-MONAI/MONAI/issues/3423
        GarbageCollector(log_level=20, trigger_event=train_trigger_event),
    ]

    # logger.info(f"{MAX_EPOCHS=}")
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=MAX_EPOCHS,
        train_data_loader=train_loader,
        network=network,
        iteration_update=Interaction(
            deepgrow_probability=args.deepgrow_probability_train,
            transforms=click_transforms,
            click_probability_key="probability",
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
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )

    save_dict = {'trainer': trainer, "net": network, "opt": optimizer, "lr": lr_scheduler}
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

    # handler = None
    if args.resume_from != "None":
        logger.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        checkpoint = torch.load(args.resume_from)
        # print(*checkpoint['trainer'].items(), sep='\n')
        
        for key in save_dict:
            assert key in checkpoint, f"key {key} has not been found in the save_dict! The file may be broken or incompatible (e.g. evaluator has not been run).\n file keys: {checkpoint.keys}"
        # network.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['opt'])
        # args.current_epoch = int(checkpoint['lr']["last_epoch"])
        # print(f"#### CURRENT EPOCH: {args.current_epoch} ######")
        # CURRENT_EPOCH = args.current_epoch
        # MAX_EPOCHS = args.epochs - CURRENT_EPOCH
        # assert MAX_EPOCHS > 0
        logger.critical("!!!!!!!!!!!!!!!!!!!! RESUMING !!!!!!!!!!!!!!!!!!!!!!!!!")
        # logger.critical(f"This code assumes that the previous run shall be continuted, so now it running from epoch {CURRENT_EPOCH} to {CURRENT_EPOCH + MAX_EPOCHS}")
        
        # lr_scheduler.load_state_dict(checkpoint['lr'])
        # logger.info(f"Resuming lr_scheduler from epoch: {lr_scheduler.last_epoch} last_lr: {lr_scheduler.get_last_lr()}")

        handler = CheckpointLoader(load_path=args.resume_from, load_dict=save_dict, map_location=map_location)
        handler(trainer)
        # exit(0)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        init_tensorboard_logger(trainer, evaluator)

    return trainer, evaluator, tb_logger


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")

    # if args.export:
    #     logger.info(
    #         "{}:: Loading PT Model from: {}".format(args.gpu, args.input)
    #     )
        
    #     network = get_network(args.network, args.labels).to(device)

    #     map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
    #     network.load_state_dict(torch.load(args.input, map_location=map_location))

    #     logger.info("{}:: Saving TorchScript Model".format(args.gpu))
    #     model_ts = torch.jit.script(network)
    #     torch.jit.save(model_ts, os.path.join(args.output))
    #     return
    
    gpu_thread = GPU_Thread(1, "Track_GPU_Usage", f"{args.output}/usage.csv", device)
    logger.info(f"Logging GPU usage to {args.output}/usage.csv")

    try:
        wp = WorkflowProfiler()
        trainer, evaluator, tb_logger = create_trainer(args)
        gpu_thread.start()
        terminator = TerminationHandler(args, tb_logger, wp, gpu_thread)
        
        with tb_logger:         
            with wp:
                start_time = time.time()
                for t, name in [(trainer, "trainer"), (evaluator, "evaluator")]:
                    for event in [["Epoch", wp, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED],
                                  ["Iteration", wp, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED],
                                  ["Batch generation", wp, Events.GET_BATCH_STARTED, Events.GET_BATCH_COMPLETED],
                                  ["Inner Iteration", wp, IterationEvents.INNER_ITERATION_STARTED, IterationEvents.INNER_ITERATION_COMPLETED],
                                  ["Whole run", wp, Events.STARTED, Events.COMPLETED],
                                   ]:
                        event[0] = f"{name}: {event[0]}"
                        # print(*event)
                        ProfileHandler(*event).attach(t)
                
                try:
                    if not args.eval_only:
                        trainer.run()
                    else:
                        evaluator.run()
                except torch.cuda.OutOfMemoryError:
                    oom_observer(device, None, None, None)
                    logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
                    
                except RuntimeError as e:
                    if "cuDNN" in str(e):
                        # Got a cuDNN error
                        pass
                    oom_observer(device, None, None, None)
                    logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
                    
                finally:
                    logger.info(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
                    logger.info("Total Training Time {}".format(time.time() - start_time))
                    

        if not args.eval_only:
            logger.info("{}:: Saving Final PT Model".format(args.gpu))
            
            torch.save(
                trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.pt")
            )

            logger.info("{}:: Saving TorchScript Model".format(args.gpu))
            model_ts = torch.jit.script(trainer.network)
            torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.ts"))
    finally:
        terminator.cleanup()
        terminator.join_threads()
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        logger.info(f"\n{wp.get_times_summary_pd()}")

def main():
    global logger
    global output_dir
    
    torch.set_num_threads(int(os.cpu_count() / 3)) # Limit number of threads to 1/3 of resources

    args, logger = parse_args()
    
    # for OOM debugging
    output_dir = args.output

    run(args)

if __name__ == "__main__":
    main()
