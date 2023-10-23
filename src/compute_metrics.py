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
from pathlib import Path

import torch
from monai.handlers import write_metrics_reports
from monai.metrics import DiceMetric, SurfaceDiceMetric
from monai.utils import string_list_all_gather

from sw_fastedit.data import get_metrics_loader, get_metrics_transforms

logger = logging.getLogger(__name__)

"""
compute_metrics.py

Computes the metrics of a labels against a prediction dir. Currently the file type is defined as "*.nii.gz" in get_metrics_loader.
"""


def run(args):
    device = torch.device(f"cuda:{args.gpu}")
    args.device = device
    args.debug = False
    args.no_log = True
    torch.cuda.set_device(device)

    data_list = get_metrics_loader(args)
    transforms = get_metrics_transforms(device=device, labels=args.labels, args=args)
    print(f"There are {len(data_list)} items in the dataloader")

    assert len(data_list) > 0

    # compute metrics for current process
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    include_background = False
    amount_of_classes = len(args.labels)
    if include_background is False:
        amount_of_classes -= 1
    class_thresholds = (0.5,) * amount_of_classes
    surface_dice_metric = SurfaceDiceMetric(
        include_background=False, class_thresholds=class_thresholds, reduction="mean", get_not_nans=False
    )
    filenames = []
    for i in data_list:
        pred_file_name = Path(i["pred"]).stem.split(".")[0]
        batchdata = transforms(i)
        batchdata["pred"] = batchdata["pred"].unsqueeze(0)
        batchdata["label"] = batchdata["label"].unsqueeze(0)

        print(f"{pred_file_name}:: pred.shape: {batchdata['pred'].shape} label.shape: {batchdata['label'].shape}")
        dice_metric(y_pred=batchdata["pred"], y=batchdata["label"])
        surface_dice_metric(y_pred=batchdata["pred"], y=batchdata["label"])
        filenames.append(pred_file_name)

    # all-gather results from all the processes and reduce for final result
    dice_results = dice_metric.aggregate().item()
    surface_dice_results = surface_dice_metric.aggregate().item()
    filenames = string_list_all_gather(strings=filenames)

    print(f"dice: {dice_results}")
    print(f"surface_dice: {surface_dice_results}")

    # generate metrics reports at: output/mean_dice_raw.csv, output/mean_dice_summary.csv, output/metrics.csv
    write_metrics_reports(
        save_dir=f"{args.output_dir}",
        images=filenames,
        metrics={"dice": dice_results, "surface_dice": surface_dice_results},
        metric_details={"dice": dice_metric.get_buffer(), "surface_dice": surface_dice_metric.get_buffer()},
        summary_ops="*",
    )

    dice_metric.reset()
    surface_dice_metric.reset()


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("-l", "--labels_dir", required=True)
    parser.add_argument("-p", "--predictions_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument(
        "-t",
        "--limit",
        type=int,
        default=0,
        help="Limit the amount of training/validation samples",
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.num_workers = 1
    args.labels = {"spleen": 1, "background": 0}

    if not os.path.exists(args.output_dir):
        pathlib.Path(args.output_dir).mkdir(parents=True)

    run(args)


if __name__ == "__main__":
    main()
