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

# Code extension and modification by M.Sc. Zdravko Marinov, Karlsuhe Institute of Techonology #
# zdravko.marinov@kit.edu #
# Further code extension and modification by B.Sc. Matthias Hadlich, Karlsuhe Institute of Techonology #
# matthiashadlich@posteo.de #

from __future__ import annotations

import os
import pathlib
import resource
import sys
import time

import pandas as pd
import torch
from ignite.engine import Events
from monai.engines.utils import IterationEvents
from monai.utils.profiling import ProfileHandler, WorkflowProfiler

from sw_interactive_segmentation.api import  oom_observer, get_test_evaluator, get_network, get_inferers, get_key_metric, get_pre_transforms
from sw_interactive_segmentation.utils.argparser import parse_args, setup_environment_and_adapt_args
from sw_interactive_segmentation.utils.tensorboard_logger import init_tensorboard_logger
from sw_interactive_segmentation.utils.helper import GPU_Thread, TerminationHandler, get_gpu_usage, handle_exception, is_docker
from sw_interactive_segmentation.data import post_process_AutoPET2_Challenge_file_list, get_test_loader, get_post_transforms_unsupervised

from monai.data import DataLoader, decollate_batch
from monai.transforms.utils import allow_missing_keys_mode


logger = None

"""
test.py

Use this file for the AutoPET2 Challenge. It does not expect any labels and works on input images only.

"""

def run(args):
    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")

    _, pre_transforms_test = get_pre_transforms(args.labels, device, args, input_keys=("image",))
    test_loader = get_test_loader(args, pre_transforms_test)

    # click_transforms = get_click_transforms(device, args)
    pred_dir = os.path.join(args.output_dir, "predictions")
    post_transform = get_post_transforms_unsupervised(args.labels, device, pred_dir=pred_dir, pretransform=pre_transforms_test)

    network = get_network(args.network, args.labels, args.non_interactive).to(device)
    _, test_inferer = get_inferers(
        args.inferer,
        args.sw_roi_size,
        args.train_crop_size,
        args.val_crop_size,
        args.train_sw_batch_size,
        args.val_sw_batch_size,
        args.sw_overlap,
        True,
    )

    val_key_metric = get_key_metric(str_to_prepend="val_")

    evaluator = get_test_evaluator(
        args,
        network=network,
        inferer=test_inferer,
        device=device,
        val_loader=test_loader,
        # loss_function=loss_function,
        # click_transforms=click_transforms,
        post_transform=post_transform,
        resume_from=args.resume_from,
        # key_val_metric=val_key_metric,
    )

    save_dict = {
        "net": network,
    }
    

    try:
        evaluator.run()
    except torch.cuda.OutOfMemoryError:
        # oom_observer(device, None, None, None)
        logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
        raise

    except RuntimeError as e:
        if "cuDNN" in str(e):
            # Got a cuDNN error
            pass
        # oom_observer(device, None, None, None)
        logger.critical(get_gpu_usage(device, used_memory_only=False, context="ERROR"))
        raise

    # POSTPROCESSING for the challenge

    if args.dataset == "AutoPET2_Challenge":
        # convert the mha to nifti
        post_process_AutoPET2_Challenge_file_list(args, pred_dir=pred_dir, cache_dir=args.cache_dir)


def main():
    global logger

    # Slurm only: Speed up the creation of temporary files
    if os.environ.get("SLURM_JOB_ID") is not None:
        tmpdir = "/local/work/mhadlich/tmp"
        os.environ["TMPDIR"] = tmpdir
        if not os.path.exists(tmpdir):
            pathlib.Path(tmpdir).mkdir(parents=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))

    sys.excepthook = handle_exception

    if not is_docker():
        torch.set_num_threads(int(os.cpu_count() / 3))  # Limit number of threads to 1/3 of resources

    args = parse_args()
    args, logger = setup_environment_and_adapt_args(args)

    run(args)


if __name__ == "__main__":
    main()
