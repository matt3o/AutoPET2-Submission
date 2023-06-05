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


# Things needed to debug the Interaction class
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8*8192, rlimit[1]))

import torch
import cupy as cp

from utils.logger import setup_loggers, get_logger

logger = None
output_dir = None

from ignite.handlers import Timer, BasicTimeProfiler, HandlersTimeProfiler
from utils.helper import print_gpu_usage, get_gpu_usage, get_actual_cuda_index_of_device, get_git_information

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

import threading

from monai.optimizers.novograd import Novograd

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



def oom_observer(device, alloc, device_alloc, device_free):
    # snapshot right after an OOM happened
    print('saving allocated state during OOM')
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open(f'{output_dir}/oom_snapshot.pickle', 'wb'))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CustomLoader:
    def __init__(self, name: Optional[str] = None):
        self._name = name

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)


    def __call__(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        self.logger.info(get_gpu_usage(engine.state.device, used_memory_only=False, context="Events.EPOCH_COMPLETED"))
        self.logger.info(torch.cuda.memory_summary())



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    oom_observer(None, None, None, None)
    logger.critical(torch.cuda.memory_summary())
    
    

sys.excepthook = handle_exception

class GPU_Thread(threading.Thread):
    def __init__(self, threadID: int, name: str, output_file: str, device: torch.device, event: threading.Event):
        super().__init__()
        self.threadID = threadID
        self.name = name
        self.device = device
        self.csv_file = open(f"{output_file}", "w")
        header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
        self.csv_file.write(header)
        self.csv_file.write("\n")
        self.csv_file.flush()
        self.stopped = event

    def __del__(self):
        self.csv_file.flush()
        self.csv_file.close()

    def run(self):
        while not self.stopped.wait(1):
            header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
            self.csv_file.write(usage)
            self.csv_file.write("\n")
            self.csv_file.flush()


def get_network(network, labels, args):
    if network == "dynunet":
        network = DynUNet(
            spatial_dims=3,
            in_channels=len(labels) + 1,
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
            in_channels=len(labels) + 1,
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

# def track_gpu_usage(output: str, device: torch.device):
#     print_gpu_usage(device)
#     get_gpu_usage(device)

class GPU_Thread(threading.Thread):
    def __init__(self, threadID: int, name: str, output_file: str, device: torch.device, event: threading.Event):
        super().__init__()
        self.threadID = threadID
        self.name = name
        self.device = device
        self.csv_file = open(f"{output_file}", "w")
        header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
        self.csv_file.write(header)
        self.csv_file.write("\n")
        self.csv_file.flush()
        self.stopped = event

    def __del__(self):
        self.csv_file.flush()
        self.csv_file.close()

    def run(self):
        while not self.stopped.wait(1):
            header, usage = get_gpu_usage(self.device, used_memory_only=False, context="", csv_format=True)
            self.csv_file.write(usage)
            self.csv_file.write("\n")
            self.csv_file.flush()



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

    # define training components
    network = get_network(args.network, args.labels, args).to(device)

    print('Number of parameters:', f"{count_parameters(network):,}")

    if args.model_filepath != 'None' and not args.resume:
        raise UserWarning("To correctly load a network you need to add --resume otherwise no model will be loaded...")
    if args.resume:
        logger.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}

        network.load_state_dict(
            torch.load(args.model_filepath, map_location=map_location)['net']
        )

    
    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        CustomLoader(),
        # CheckpointSaver(
        #     save_dir=args.output,
        #     save_dict={"net": network},
        #     save_key_metric=True,
        #     save_final=True,
        #     save_interval=args.save_interval,
        #     final_filename="pretrained_deepedit_" + args.network + ".pt",
        # ),
    ]
    

    all_val_metrics = dict()
    all_val_metrics["val_mean_dice"] = MeanDice(
        output_transform=from_engine(["pred", "label"]), include_background=False
    )
    for key_label in args.labels:
        if key_label != "background":
            all_val_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    if args.inferer == "SimpleInferer":
        inferer=SimpleInferer()
        train_inferer = eval_inferer = inferer
    elif args.inferer == "SlidingWindowInferer":
        train_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=1, overlap=0)
        eval_inferer = SlidingWindowInferer(roi_size=args.sw_roi_size, sw_batch_size=1, mode="gaussian")
    else:
        raise UserWarning("Invalid Inferer selected")

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
        ),
        inferer=eval_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)#, squared_pred=True) #,batch=True)
    
    if args.novograd:
        optimizer = Novograd(network.parameters(), args.learning_rate)
    else: # default
        optimizer = torch.optim.Adam(network.parameters(), args.learning_rate)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator, interval=args.val_freq, epoch_level=(not args.eval_only)
        ),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
        TensorBoardStatsHandler(
            log_dir=args.output,
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network, "opt": optimizer, "lr": lr_scheduler},
            save_interval=args.save_interval * 2,
            save_final=True,
            final_filename="checkpoint.pt",
        ),
        CustomLoader(),
    ]
    

    all_train_metrics = dict()
    all_train_metrics["train_dice"] = MeanDice(output_transform=from_engine(["pred", "label"]),
                                               include_background=False)
    for key_label in args.labels:
        if key_label != "background":
            all_train_metrics[key_label + "_dice"] = MeanDice(
                output_transform=from_engine(["pred_" + key_label, "label_" + key_label]), include_background=False
            )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
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
        ),
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=train_inferer,
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )

    return trainer, evaluator


def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    if args.export:
        logger.info(
            "{}:: Loading PT Model from: {}".format(args.gpu, args.input)
        )
        device = torch.device(f"cuda:{args.gpu}")
        network = get_network(args.network, args.labels).to(device)

        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        network.load_state_dict(torch.load(args.input, map_location=map_location))

        logger.info("{}:: Saving TorchScript Model".format(args.gpu))
        model_ts = torch.jit.script(network)
        torch.jit.save(model_ts, os.path.join(args.output))
        return

    if not os.path.exists(args.output):
        logger.info(
            "output path [{}] does not exist. creating it now.".format(args.output)
        )
        os.makedirs(args.output, exist_ok=True)

    # Init the Inferer
    args.sw_roi_size = eval(args.sw_roi_size)
    args.crop_spatial_size = eval(args.crop_spatial_size)

    # verify both have a valid size (for Unet with seven layers)
    assert len(args.sw_roi_size) == 3 and len(args.crop_spatial_size) == 3
    if args.network == "dynunet":
        for size in args.crop_spatial_size:
            assert (size % 64) == 0
    
    # click-generation
    logger.warning("click_generation: This has not been implemented, so the value '{}' will be discarded for now!".format(args.click_generation))

    stopFlag = threading.Event()
    gpu_thread = GPU_Thread(1, "Track_GPU_Usage", f"{args.output}/usage.csv", torch.device(f"cuda:{args.gpu}"), stopFlag)
    logger.info(f"Logging GPU usage to {args.output}/usage.csv")
    # TODO test this on torch v2 
    # torch.set_default_device(f"cuda:{args.gpu}")

    try:
        wp = WorkflowProfiler()
        
        with wp:           
            trainer, evaluator = create_trainer(args)
            gpu_thread.start()

            epoch_h = ProfileHandler("Epoch", wp, Events.EPOCH_STARTED, Events.EPOCH_COMPLETED).attach(trainer)
            iter_h = ProfileHandler("Iteration", wp, Events.ITERATION_STARTED, Events.ITERATION_COMPLETED).attach(trainer)
            batch_h = ProfileHandler("Batch generation", wp, Events.GET_BATCH_STARTED, Events.GET_BATCH_COMPLETED).attach(trainer)
            innter_iteration_h = ProfileHandler("Inner Iteration", wp, IterationEvents.INNER_ITERATION_STARTED, IterationEvents.INNER_ITERATION_COMPLETED).attach(trainer)
            whole_run_h = ProfileHandler("Whole run", wp, Events.STARTED, Events.COMPLETED).attach(trainer)

            start_time = time.time()
            # torch._C._cuda_attach_out_of_memory_observer(cb)
            
            try:
                if not args.eval_only:
                    trainer.run()
                else:
                    evaluator.run()
            except torch.cuda.OutOfMemoryError:
                oom_observer(None, None, None, None)
                logger.critical(get_gpu_usage(torch.device(f"cuda:{args.gpu}"), used_memory_only=False, context="ERROR"))
                
            except RuntimeError as e:
                if "cuDNN" in str(e):
                    # Got a cuDNN error
                    pass
                oom_observer(None, None, None, None)
                logger.critical(get_gpu_usage(torch.device(f"cuda:{args.gpu}"), used_memory_only=False, context="ERROR"))
                
            finally:
                stopFlag.set()
                logger.info(get_gpu_usage(torch.device(f"cuda:{args.gpu}"), used_memory_only=False, context="ERROR"))
                logger.info("Total Training Time {}".format(time.time() - start_time))
                logger.info(f"\n{wp.get_times_summary_pd()}")
                gpu_thread.join()

        if not args.eval_only:
            logger.info("{}:: Saving Final PT Model".format(args.gpu))
            
            torch.save(
                trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.pt")
            )

            logger.info("{}:: Saving TorchScript Model".format(args.gpu))
            model_ts = torch.jit.script(trainer.network)
            torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.ts"))
    finally:
        # Cleanup
        if args.throw_away_cache:
            logger.info("Cleaning up..")
            shutil.rmtree(args.cache_dir, ignore_errors=True)
        else:
            logger.info("Leaving cache dir as it is..")



def main():
    global logger
    global output_dir
    torch.cuda.init()
    torch.cuda.memory._record_memory_history(True)
    #torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    
    torch.set_num_threads(int(os.cpu_count() / 3)) # Limit number of threads to 1/3 of resources
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input", default="/cvhci/data/AutoPET/AutoPET/")
    parser.add_argument("-o", "--output", default="/cvhci/temp/mhadlich/output")
    parser.add_argument("-d", "--data", default="/cvhci/temp/mhadlich/data")
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
    parser.add_argument("--crop_spatial_size", default="(128,128,128)", action='store')

    # Training
    parser.add_argument("-a", "--amp", default=False, action='store_true')
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("--novograd", default=False, action='store_true')
    parser.add_argument("--model_weights", type=str, default='None')
    parser.add_argument("--best_val_weights", default=False, action='store_true')

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
    parser.add_argument("--debug", default=False, action='store_true')

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

    # Restoring previous model if resume flag is True
    args.model_filepath = args.model_weights
    if args.best_val_weights:
        args.model_filepath = os.path.join(args.output, sorted([el for el in os.listdir(args.output) if 'net_key' in el])[-1])
        current_epoch = sorted([int(el.split('.')[0].split('=')[1]) for el in os.listdir(args.output) if 'net_epoch' in el])[-1]
        args.epochs = args.epochs - current_epoch # Reset epochs based on previous model

    if not os.path.exists(args.output):
        pathlib.Path(args.output).mkdir(parents=True)
    
    output_dir = args.output
    
    if args.throw_away_cache:
        args.cache_dir = f"{args.cache_dir}/{uuid.uuid4()}"
    else:
        args.cache_dir = f"{args.cache_dir}"

    if not os.path.exists(args.cache_dir):
        pathlib.Path(args.cache_dir).mkdir(parents=True)
    
    if not os.path.exists(args.data):
        pathlib.Path(args.data).mkdir(parents=True)

    args.real_cuda_device = get_actual_cuda_index_of_device(torch.device(f"cuda:{args.gpu}"))

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
    logger.info(f"CPU Count: {os.cpu_count()}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    run(args)

if __name__ == "__main__":
    main()
