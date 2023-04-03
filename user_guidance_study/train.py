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

# Things needed to debug the Interaction class
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
from utils.interaction import Interaction

from utils.utils import get_pre_transforms, get_click_transforms, get_post_transforms, get_loaders

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
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss
from utils.dynunet import DynUNet

from monai.utils import set_determinism
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_network(network, labels, args):
    network = DynUNet(
        spatial_dims=3,
        in_channels=len(labels) + 1,
        out_channels=len(labels),
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, [2, 2, 1]],
        upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
        conv1d=args.conv1d,
        conv1s=args.conv1s,
    )
    return network


def create_trainer(args):

    set_determinism(seed=args.seed)

    device = torch.device(f"cuda:{args.gpu}")

    pre_transforms_train, pre_transforms_val = get_pre_transforms(args.labels, device, args)
    click_transforms = get_click_transforms(device, args)
    post_transform = get_post_transforms(args.labels)

    train_loader, val_loader = get_loaders(args, pre_transforms_train, pre_transforms_val)

    # define training components
    network = get_network(args.network, args.labels, args).to(device)

    print('Number of parameters:', f"{count_parameters(network):,}")

    if args.resume:
        logging.info("{}:: Loading Network...".format(args.gpu))
        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}

        network.load_state_dict(
            torch.load(args.model_filepath, map_location=map_location)['net']
        )

    # define event-handlers for engine
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir=args.output, output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=args.output,
            save_dict={"net": network},
            save_key_metric=True,
            save_final=True,
            save_interval=args.save_interval,
            final_filename="pretrained_deepedit_" + args.network + ".pt",
        ),
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
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        key_val_metric=all_val_metrics,
        val_handlers=val_handlers,
    )

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
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
        inferer=SimpleInferer(),
        postprocessing=post_transform,
        amp=args.amp,
        key_train_metric=all_train_metrics,
        train_handlers=train_handlers,
    )
    return trainer, evaluator


def run(args):
    for arg in vars(args):
        logging.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")

    if args.export:
        logging.info(
            "{}:: Loading PT Model from: {}".format(args.gpu, args.input)
        )
        device = torch.device(f"cuda:{args.gpu}")
        network = get_network(args.network, args.labels).to(device)

        map_location = {f"cuda:{args.gpu}": "cuda:{}".format(args.gpu)}
        network.load_state_dict(torch.load(args.input, map_location=map_location))

        logging.info("{}:: Saving TorchScript Model".format(args.gpu))
        model_ts = torch.jit.script(network)
        torch.jit.save(model_ts, os.path.join(args.output))
        return

    if not os.path.exists(args.output):
        logging.info(
            "output path [{}] does not exist. creating it now.".format(args.output)
        )
        os.makedirs(args.output, exist_ok=True)

    trainer, evaluator = create_trainer(args)

    start_time = time.time()
    trainer.run()
    end_time = time.time()

    logging.info("Total Training Time {}".format(end_time - start_time))
    logging.info("{}:: Saving Final PT Model".format(args.gpu))
    torch.save(
        trainer.network.state_dict(), os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.pt")
    )

    logging.info("{}:: Saving TorchScript Model".format(args.gpu))
    model_ts = torch.jit.script(trainer.network)
    torch.jit.save(model_ts, os.path.join(args.output, "pretrained_deepedit_" + args.network + "-final.ts"))


def main():
    print(f"CPU Count: {os.cpu_count()}")
    torch.set_num_threads(int(os.cpu_count() / 3)) # Limit number of threads to 1/3 of resources
    print(f"Num threads: {torch.get_num_threads()}")
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input", default="../../AutoPET_deepedit/AutoPET")
    parser.add_argument("-o", "--output", default="output/test")
    parser.add_argument("--cache_dir", type=str, default='cache/test')
    parser.add_argument("-x", "--split", type=float, default=0.8)
    parser.add_argument("-t", "--limit", type=int, default=0, help='Limit the amount of training/validation samples')


    # Configuration
    parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)

    # Model
    parser.add_argument("-n", "--network", default="dynunet", choices=["dynunet"])
    parser.add_argument("-r", "--resume", default=False, action='store_true')

    # Training
    parser.add_argument("-a", "--amp", default=False, action='store_true')
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
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

    # Guidance Signal Hyperparameters
    parser.add_argument("--sigma", type=int, default=3)
    parser.add_argument("--disks", default=False, action='store_true')
    parser.add_argument("--edt", default=False, action='store_true')
    parser.add_argument("--gdt", default=False, action='store_true')
    parser.add_argument("--gdt_th", type=float, default=10)
    parser.add_argument("--exp_geos", default=False, action='store_true')
    parser.add_argument("--conv1d", default=False, action='store_true')
    parser.add_argument("--conv1s", default=False, action='store_true')
    parser.add_argument("--adaptive_sigma", default=False, action='store_true')

    parser.add_argument("--dataset", default="MSD_Spleen")

    args = parser.parse_args()
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
        os.mkdir(args.output)
    if not os.path.exists(args.cache_dir):
        os.mkdir(args.cache_dir)
    run(args)


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
