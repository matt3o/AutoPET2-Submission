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

from __future__ import annotations

import os
import pathlib
import resource
import sys
import time
import argparse

import pandas as pd
import torch
from ignite.engine import Events
from monai.engines.utils import IterationEvents
from monai.utils.profiling import ProfileHandler, WorkflowProfiler

from sw_interactive_segmentation.api import get_trainer, oom_observer
from sw_interactive_segmentation.argparser import (
    setup_environment_and_adapt_args)
from sw_interactive_segmentation.tensorboard_logger import \
    init_tensorboard_logger
from sw_interactive_segmentation.utils.helper import (GPU_Thread,
                                                      TerminationHandler,
                                                      get_gpu_usage,
                                                      handle_exception)
from sw_interactive_segmentation import get_tester
from sw_interactive_segmentation.utils.utils import get_test_loader

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    ScaleIntensityd,
    ToDeviced,
)
from monai.metrics import DiceMetric
from monai.utils import string_list_all_gather
from monai.handlers import write_metrics_reports

# Various settings #

logger = None


def run(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    loader = get_test_loader(args)

    # define transforms for predictions and labels
    transforms = Compose(
        [
            LoadImaged(keys=["pred", "label"]),
            ToDeviced(keys=["pred", "label"], device=device),
            EnsureChannelFirstd(keys=["pred", "label"]),
        ]
    )
    data_part = [transforms(item) for item in loader]

    # compute metrics for current process
    metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    metric(y_pred=[i["pred"] for i in data_part], y=[i["label"] for i in data_part])
    filenames = [item["pred_meta_dict"]["filename_or_obj"] for item in data_part]
    # all-gather results from all the processes and reduce for final result
    result = metric.aggregate().item()
    filenames = string_list_all_gather(strings=filenames)

    print("mean dice: ", result)
    # generate metrics reports at: output/mean_dice_raw.csv, output/mean_dice_summary.csv, output/metrics.csv
    write_metrics_reports(
        save_dir=f"{args.output_dir}",
        images=filenames,
        metrics={"mean_dice": result},
        metric_details={"mean_dice": metric.get_buffer()},
        summary_ops="*",
    )

    metric.reset()


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-i", "--input_dir", default="/cvhci/data/AutoPET/AutoPET/")
    parser.add_argument("-l", "--labels_dir", default="None")
    parser.add_argument("-p", "--predictions_dir", default="None")
    parser.add_argument("-o", "--output_dir", default="/cvhci/temp/mhadlich/output")

    # parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    global logger

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tmpdir = "/local/work/mhadlich/tmp"
    if os.environ.get("SLURM_JOB_ID") is not None:
        os.environ["TMPDIR"] = tmpdir
        if not os.path.exists(tmpdir):
            pathlib.Path(tmpdir).mkdir(parents=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))

    sys.excepthook = handle_exception

    torch.set_num_threads(
        int(os.cpu_count() / 3)
    )  # Limit number of threads to 1/3 of resources

    args = parse_args()
    # args, logger = setup_environment_and_adapt_args(args)

    if not os.path.exists(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True)

    run(args)


if __name__ == "__main__":
    main()
