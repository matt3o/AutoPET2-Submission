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


import argparse
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

# tmpdir = '/local/work/mhadlich/tmp'
# if os.environ.get("SLURM_JOB_ID") is not None:
#     os.environ['TMPDIR'] = tmpdir
#     if not os.path.exists(tmpdir):
#         pathlib.Path(tmpdir).mkdir(parents=True)

# Things needed to debug the Interaction class
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8*8192, rlimit[1]))

import pandas as pd
import torch
import cupy as cp

from utils.logger import setup_loggers, get_logger

logger = None
output_dir = None

from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler
from utils.helper import print_gpu_usage, get_gpu_usage, get_actual_cuda_index_of_device, get_git_information, gpu_usage

from utils.interaction import Interaction
from utils.utils import get_pre_transforms, get_click_transforms, get_post_transforms, get_loaders

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
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from utils.dynunet import DynUNet

from monai.utils.profiling import ProfileHandler, WorkflowProfiler
from monai.engines.utils import IterationEvents

from ignite.engine import Engine, Events
import threading
from monai.data import set_track_meta
from monai.utils import set_determinism
from ignite.handlers import TerminateOnNan

import threading

from monai.optimizers.novograd import Novograd

from ignite.contrib.handlers.tensorboard_logger import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def oom_observer(device, alloc, device_alloc, device_free):
    if device is not None and logger is not None:
        logger.critical(torch.cuda.memory_summary(device))
    # snapshot right after an OOM happened
    print('saving allocated state during OOM')
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f'{output_dir}/oom_snapshot.pickle', 'wb'))
    # logger.critical(snapshot)
    torch.cuda.memory._save_memory_usage(filename=f"{output_dir}/memory.svg", snapshot=snapshot)
    torch.cuda.memory._save_segment_usage(filename=f"{output_dir}/segments.svg", snapshot=snapshot)


class TerminationHandler:
    def __init__(self, args, tb_logger, wp, gpu_thread):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.args = args
        self.tb_logger = tb_logger
        self.wp = wp
        self.gpu_thread = gpu_thread

    def exit_gracefully(self, *args):
        logger.critical(f"#### RECEIVED TERM SIGNAL - ABORTING RUN ############")
        self.cleanup()
        self.join_threads()
        logger.info(f"\n{self.wp.get_times_summary_pd()}")
        sys.exit(99)

    def join_threads(self):
        self.tb_logger.close()
        self.gpu_thread.stopFlag.set()
        self.gpu_thread.join()

    def cleanup(self):
        logger.info(f"#### LOGGED ALL DATA TO {self.args.output} ############")
        # Cleanup
        if self.args.throw_away_cache:
            logger.info("Cleaning up..")
            shutil.rmtree(self.args.cache_dir, ignore_errors=True)
        else:
            logger.info("Leaving cache dir as it is..")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    logger.critical(torch.cuda.memory_summary())
    
    

sys.excepthook = handle_exception

class GPU_Thread(threading.Thread):
    def __init__(self, threadID: int, name: str, output_file: str, device: torch.device):#, event: threading.Event):
        super().__init__(daemon=True)
        self.threadID = threadID
        self.name = name
        self.device = device
        self.csv_file = open(f"{output_file}", "w")
        header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
        self.csv_file.write(header)
        self.csv_file.write("\n")
        self.csv_file.flush()
        #self.stopped = event
        self.stopFlag = threading.Event()

    def __del__(self):
        self.csv_file.flush()
        self.csv_file.close()

    def run(self):
        while not self.stopFlag.wait(1):
            header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
            self.csv_file.write(usage)
            self.csv_file.write("\n")
            self.csv_file.flush()


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
            conv1d=args.conv1d,
            conv1s=args.conv1s,
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
            conv1d=args.conv1d,
            conv1s=args.conv1s,
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
        train_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=train_batch_size, mode="gaussian")
        eval_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=val_batch_size, mode="gaussian")

    # OPTIMIZER
    if args.optimizer == "Novograd":
        optimizer = Novograd(network.parameters(), args.learning_rate)
    elif args.optimizer == "Adam": # default
        optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)

    MAX_EPOCHS = args.epochs
    CURRENT_EPOCH = args.current_epoch

    # SCHEDULER
    if args.scheduler == "MultiStepLR":
        steps = 4
        steps_per_epoch = round(MAX_EPOCHS/steps)
        if steps_per_epoch < 1:
            logger.error("Chosen number of epochs {MAX_EPOCHS}/{steps} < 0")
            milestones= range(0, MAX_EPOCHS)
        else:
            milestones = [num for num in range(0, MAX_EPOCHS) if num % round(steps_per_epoch) == 0][1:]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.333, last_epoch=CURRENT_EPOCH)
    elif args.scheduler == "PolynomialLR":
        lr_scheduler =  torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=MAX_EPOCHS, power = 2, last_epoch=CURRENT_EPOCH)
    elif args.scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min = 1e-6, last_epoch=CURRENT_EPOCH)

    
    # ckpt_loader = None
    if args.model_filepath != 'None' and not args.resume:
        raise UserWarning("To correctly load a network you need to add --resume otherwise no model will be loaded...")
    
    if args.resume:
        logger.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        checkpoint = torch.load(args.model_filepath, map_location=map_location)
        network.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['opt'])
        args.current_epoch = int(checkpoint['lr']["last_epoch"])
        CURRENT_EPOCH = args.current_epoch
        MAX_EPOCHS = MAX_EPOCHS - CURRENT_EPOCH
        assert MAX_EPOCHS > 0
        logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.critical(f"This code assumes that the previous run shall be continuted, so now it running from epoch {CURRENT_EPOCH} to {MAX_EPOCHS}")
        
        lr_scheduler.load_state_dict(checkpoint['lr'])
        logger.info(f"Resuming lr_scheduler from epoch: {lr_scheduler.last_epoch} last_lr: {lr_scheduler.get_last_lr()}")
        
        


        
    train_trigger_event = Events.ITERATION_COMPLETED(every=200) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=50)
    val_trigger_event = Events.ITERATION_COMPLETED(every=50) if args.gpu_size == "large" else Events.ITERATION_COMPLETED(every=1)
    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
             save_dir=args.output,
             save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
             save_key_metric=True,
             save_final=True,
             save_interval=args.save_interval,
             final_filename="pretrained_deepedit_" + args.network + ".pt",
        ),

        # TensorBoardStatsHandler(log_dir=args.output, iteration_log=False, output_transform=lambda x: None, global_epoch_transform=lambda x: trainer.state.epoch),
        # CustomLoader(),
        # https://github.com/Project-MONAI/MONAI/issues/3423
        GarbageCollector(log_level=20, trigger_event=val_trigger_event),
    ]

    all_val_metrics = dict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    # Disabled since it led to weird artefacts in the Tensorboard diagram
    # for key_label in args.labels:
    #     if key_label != "background":
    #         all_val_metrics[key_label + "_dice"] = MeanDice(
    #             output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
    #         )

    # squared_pred enables much faster convergence, possibly even better results in the long run
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)


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
        ),
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )


    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]),
                                               include_background=False)
    if len(args.labels) > 2:
        for key_label in args.labels:
            if key_label != "background":
                all_train_metrics[key_label + "_dice"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
                )
                all_train_metrics[key_label + "_dice_with_bg"] = MeanDice(
                    output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=True
                )

    tb_logger = TensorboardLogger(log_dir=f"{args.output}/tensorboard")


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
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
            save_interval=args.save_interval,
            save_final=True,
            final_filename="checkpoint.pt",
            n_saved=2,
        ),
        # CustomLoader(all_train_metrics),
        # https://github.com/Project-MONAI/MONAI/issues/3423
        GarbageCollector(log_level=20, trigger_event=train_trigger_event),
    ]


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
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="1_validation",
        metric_names=list(all_val_metrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="2_training",
        metric_names=list(all_train_metrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="2_training",
        output_transform=lambda x: x[0]["loss"],
    )

    tb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=optimizer,
        tag="3_params"
    )

    # for debugging 
    
    # # Attach the logger to the trainer to log model's weights norm after each iteration
    # tb_logger.attach(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     log_handler=WeightsScalarHandler(network)
    # )

    # # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    # tb_logger.attach(
    #     trainer,
    #     event_name=Events.EPOCH_COMPLETED,
    #     log_handler=WeightsHistHandler(network)
    # )

    # # Attach the logger to the trainer to log model's gradients norm after each iteration
    # tb_logger.attach(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     log_handler=GradsScalarHandler(network)
    # )

    # # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    # tb_logger.attach(
    #     trainer,
    #     event_name=Events.EPOCH_COMPLETED,
    #     log_handler=GradsHistHandler(network)
    # )


    return trainer, evaluator, tb_logger


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")
    cuda_index, util_gpu, util_memory, nv_total, nv_free, nv_used, torch_reserved, cupy_usage = gpu_usage(device, used_memory_only=False)
    if nv_total < 25000:
        args.gpu_size = "small"
    elif nv_total < 55000:
        args.gpu_size = "medium"
    else:
        args.gpu_size = "large"
    logger.info(f"Selected gpu size: {args.gpu_size}")

    if args.export:
        logger.info(
            "{}:: Loading PT Model from: {}".format(args.gpu, args.input)
        )
        
        network = get_network(args.network, args.labels).to(device)

        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        network.load_state_dict(torch.load(args.input, map_location=map_location))

        logger.info("{}:: Saving TorchScript Model".format(args.gpu))
        model_ts = torch.jit.script(network)
        torch.jit.save(model_ts, os.path.join(args.output))
        return

    # Init the Inferer
    args.sw_roi_size = eval(args.sw_roi_size)
    assert len(args.sw_roi_size) == 3

    if args.val_crop_size == "None":
        args.val_crop_size = None
    else:
        args.val_crop_size = eval(args.val_crop_size)
        assert len(args.val_crop_size) == 3

    if args.train_crop_size == "None":
        args.train_crop_size = None
    else:
        args.train_crop_size = eval(args.train_crop_size)
        assert len(args.train_crop_size) == 3

    # verify both have a valid size (for Unet with seven layers)
     
    if args.network == "dynunet" and args.inferer == "SimpleInferer":
        for size in args.train_crop_size:
            assert (size % 64) == 0
        for size in args.val_crop_size:
            assert (size % 64) == 0


    # click-generation
    logger.warning("click_generation: This has not been implemented, so the value '{}' will be discarded for now!".format(args.click_generation))

    
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
        pd.set_option('display.max_columns', None)
        pd.set_option("display.max_rows", None)
        logger.info(f"\n{wp.get_times_summary_pd()}")

def main():
    global logger
    global output_dir
    
    torch.set_num_threads(int(os.cpu_count() / 3)) # Limit number of threads to 1/3 of resources
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input", default="/cvhci/data/AutoPET/AutoPET/")
    parser.add_argument("-o", "--output", default="/cvhci/temp/mhadlich/output")
    parser.add_argument("-d", "--data", default="None")
    # a subdirectory is created below cache_dir for every run
    parser.add_argument("-c", "--cache_dir", type=str, default='/cvhci/temp/mhadlich/cache')
    parser.add_argument("-ta", "--throw_away_cache", default=False, action='store_true')
    parser.add_argument("-x", "--split", type=float, default=0.8)
    parser.add_argument("-t", "--limit", type=int, default=0, help='Limit the amount of training/validation samples')

    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)

    # Model
    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet", "smalldynunet"])
    parser.add_argument("-r", "--resume", default=False, action='store_true')
    parser.add_argument("-in", "--inferer", default="SimpleInferer", choices=["SimpleInferer", "SlidingWindowInferer"])
    parser.add_argument("--sw_roi_size", default="(128,128,128)", action='store')
    parser.add_argument("--train_crop_size", default="(224,224,224)", action='store')
    parser.add_argument("--val_crop_size", default="None", action='store')
    # 1 on 24 Gb, 8 on 50 Gb,
    parser.add_argument("--train_sw_batch_size", type=int, default=8)
    parser.add_argument("--val_sw_batch_size", type=int, default=1)
    # parser.add_argument("--sw_batch_limit", type=int, default=-1)
    

    # Training
    parser.add_argument("-a", "--amp", default=False, action='store_true')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    # If learning rate is set to 0.001, the DiceCE will produce Nans?!?
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Novograd"])
    parser.add_argument("--scheduler", default="MultiStepLR", choices=["MultiStepLR", "PolynomialLR", "CosineAnnealingLR"])
    parser.add_argument("--model_weights", type=str, default='None')
    # parser.add_argument("--best_val_weights", default=False, action='store_true')

    # Logging
    parser.add_argument("-f", "--val_freq", type=int, default=1) # Epoch Level
    parser.add_argument("--save_interval", type=int, default=3)
    parser.add_argument("--export", default=False, action='store_true')
    parser.add_argument("--eval_only", default=False, action='store_true')
    parser.add_argument("--save_nifti", default=False, action='store_true')

    # Interactions
    parser.add_argument("-it", "--max_train_interactions", type=int, default=10)
    parser.add_argument("-iv", "--max_val_interactions", type=int, default=10)
    parser.add_argument("-dpt", "--deepgrow_probability_train", type=float, default=1.0)
    parser.add_argument("-dpv", "--deepgrow_probability_val", type=float, default=1.0)

    # Guidance Signal Click Generation
    parser.add_argument("-cg", "--click_generation", default="non-corrective,random", 
        choices=[
        "non-corrective,random", # Sample a random pixel from the training patch
        "corrective,global,random", # Extract all missclassified pixels, then sample one (from the whole volume), stop if probability (0.9) of stopping based on clicks (binomial distribution?)
        "corrective,global,low dice", # Extract all missclassified pixels. While the dice score remains low (threshold 0.9 or 0.95), 
        # continue sampling new clicks (stop at number of clicks to avoid infinite loops)
        "corrective,global,hybrid low dice", # Same as 'corrective,global,random' but decide whether to sample new clicks
        #  (based on the distribution on the dice, )
        # ONLY relevant during evalution 
        "corrective,patch-based,random", # Same as 'corrective,global,random' but only select those patches where the model was actually wrong (selection probability p*)
        "corrective,patch-based,low dice", # Same as 'corrective,global,low dice' but do it per patch until below threshold or at click limit
        "corrective,patch-based,hybrid low dice", # Same as 'corrective,global,hybrid low dice' but do the sampling on the patch with the lowest dice
        # top 10 worst patches
        ])

    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=1)
    parser.add_argument("--disks", default=False, action='store_true')
    parser.add_argument("--edt", default=False, action='store_true')
    parser.add_argument("--gdt", default=False, action='store_true')
    parser.add_argument("--gdt_th", type=float, default=10)
    parser.add_argument("--exp_geos", default=False, action='store_true')
    parser.add_argument("--conv1d", default=False, action='store_true')
    parser.add_argument("--conv1s", default=False, action='store_true')
    parser.add_argument("--adaptive_sigma", default=False, action='store_true')

    parser.add_argument("--no_log", default=False, action='store_true')
    parser.add_argument("--dont_check_output_dir", default=False, action='store_true')
    parser.add_argument("--debug", default=False, action='store_true')
    # small: 24, medium: 50 Gb, large 80 Gb
    # parser.add_argument("--gpu_size", default="small", choices=["small", "medium", "large"])

    parser.add_argument("--dataset", default="AutoPET") #MSD_Spleen

    # Set up additional information concerning the environment and the way the script was called
    args = parser.parse_args()
    args.caller_args = sys.argv
    args.env = os.environ
    args.git = get_git_information()

    # For single label using one of the Medical Segmentation Decathlon
    args.labels = {'spleen': 1,
                   'background': 0
                   }

    args.model_filepath = args.model_weights
    args.current_epoch = -1
    
    if not args.dont_check_output_dir and os.path.isdir(args.output):
        raise UserWarning(f"output path {args.output} already exists. Please choose another path..")
    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True)
    
    # for OOM debugging
    output_dir = args.output

    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    if args.no_log:
        log_folder_path = None
    else:
        log_folder_path = args.output
    setup_loggers(loglevel, log_folder_path)
    logger = get_logger()

    if args.throw_away_cache:
        args.cache_dir = f"{args.cache_dir}/{uuid.uuid4()}"
    else:
        args.cache_dir = f"{args.cache_dir}"

    if not os.path.exists(args.cache_dir):
        pathlib.Path(args.cache_dir).mkdir(parents=True)
    
    if args.data == "None":
        args.data = f"{args.output}/data"
        logger.info(f"--data was None, so that {args.data} was selected instead")
        
    if not os.path.exists(args.data):
        pathlib.Path(args.data).mkdir(parents=True)

    # if args.gpu_size == "24":
    #     args.sw_batch_limit = 1

    args.real_cuda_device = get_actual_cuda_index_of_device(torch.device(f"cuda:{args.gpu}"))

    logger.info(f"CPU Count: {os.cpu_count()}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    args.cwd = os.getcwd()


    run(args)

if __name__ == "__main__":
    main()
