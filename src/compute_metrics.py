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

import argparse
import logging
import os
import pathlib
import resource
from pathlib import Path

import torch
from monai.handlers import write_metrics_reports
from monai.metrics import DiceMetric
from monai.utils import string_list_all_gather

from sw_interactive_segmentation.utils.utils import get_test_loader, get_test_transforms

logger = logging.getLogger(__name__)


def run(args):
    device = torch.device(f"cuda:{args.gpu}")
    args.device = device
    torch.cuda.set_device(device)

    data_list = get_test_loader(args)
    transforms = get_test_transforms(device=device, labels=args.labels)
    print(f"There are {len(data_list)} items in the dataloader")

    assert len(data_list) > 0

    # compute metrics for current process
    metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    filenames = []
    for i in data_list:
        pred_file_name = Path(i["pred"]).stem.split(".")[0]
        batchdata = transforms(i)
        print(f"{pred_file_name}:: pred.shape: {batchdata['pred'].shape}")
        metric(y_pred=batchdata["pred"].unsqueeze(0), y=batchdata["label"].unsqueeze(0))
        filenames.append(pred_file_name)
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
    # parser.add_argument("-i", "--input_dir", default="/cvhci/data/AutoPET/AutoPET/")
    parser.add_argument("-l", "--labels_dir")
    parser.add_argument("-p", "--predictions_dir")
    parser.add_argument("-o", "--output_dir")
    # parser.add_argument("--dataset", default="AutoPET")
    parser.add_argument(
        "-t",
        "--limit",
        type=int,
        default=0,
        help="Limit the amount of training/validation samples",
    )

    # parser.add_argument("-s", "--seed", type=int, default=36)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # set_track_meta(False)

    tmpdir = "/local/work/mhadlich/tmp"
    if os.environ.get("SLURM_JOB_ID") is not None:
        os.environ["TMPDIR"] = tmpdir
        if not os.path.exists(tmpdir):
            pathlib.Path(tmpdir).mkdir(parents=True)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8 * 8192, rlimit[1]))

    torch.set_num_threads(int(os.cpu_count() / 3))  # Limit number of threads to 1/3 of resources

    args = parse_args()
    args.num_workers = 1
    args.labels = {"spleen": 1, "background": 0}

    if not os.path.exists(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True)

    run(args)


if __name__ == "__main__":
    main()