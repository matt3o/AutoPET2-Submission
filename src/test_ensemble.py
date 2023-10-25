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

import logging
import os
import resource
import sys

import torch

from sw_fastedit.api import (
    get_ensemble_evaluator,
    get_inferers,
    get_network,
    get_test_evaluator,
)
from sw_fastedit.data import get_pre_transforms
from sw_fastedit.data import (
    get_post_ensemble_transforms,
    get_post_transforms_unsupervised,
    get_test_loader,
    post_process_AutoPET2_Challenge_file_list,
)
from sw_fastedit.utils.argparser import parse_args, setup_environment_and_adapt_args
from sw_fastedit.utils.helper import handle_exception


logger = logging.getLogger("sw_fastedit")

"""
test_ensemble.py

Use this file for the AutoPET2 Challenge. It does not expect any labels and works on input images only.
"""

def run_ensemble(args):
    args.nfolds = 5

    for arg in vars(args):
        logger.info("USING:: {} = {}".format(arg, getattr(args, arg)))
    print("")
    device = torch.device(f"cuda:{args.gpu}")

    _, pre_transforms_test = get_pre_transforms(args.labels, device, args, input_keys=("image",))
    test_loader = get_test_loader(args, pre_transforms_test)

    pred_dir = os.path.join(args.output_dir, "predictions")
    post_transform = get_post_ensemble_transforms(
        labels=args.labels, device=device, pred_dir=pred_dir, pretransform=pre_transforms_test, nfolds=args.nfolds
    )

    networks = []
    for _ in range(args.nfolds):
        networks.append(get_network(args.network, args.labels, args.non_interactive).to(device))
    assert len(networks) == args.nfolds

    _, test_inferer = get_inferers(
        args.inferer,
        sw_roi_size=args.sw_roi_size,
        train_crop_size=args.train_crop_size,
        val_crop_size=args.val_crop_size,
        train_sw_batch_size=args.train_sw_batch_size,
        val_sw_batch_size=args.val_sw_batch_size,
        train_sw_overlap=args.train_sw_overlap,
        val_sw_overlap=args.val_sw_overlap,
        cache_roi_weight_map=True,
    )

    evaluator = get_ensemble_evaluator(
        args,
        networks=networks,
        inferer=test_inferer,
        device=device,
        val_loader=test_loader,
        post_transform=post_transform,
        resume_from=args.resume_from,
        nfolds=args.nfolds,
    )

    evaluator.run()

    # POSTPROCESSING for the challenge
    if args.dataset == "AutoPET2_Challenge":
        # convert the mha to nifti
        post_process_AutoPET2_Challenge_file_list(args, pred_dir=pred_dir, cache_dir=args.cache_dir)


def main():
    global logger

    args = parse_args()
    args, logger = setup_environment_and_adapt_args(args)

    run_ensemble(args)

if __name__ == "__main__":
    main()
