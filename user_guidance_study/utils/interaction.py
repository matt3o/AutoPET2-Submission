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

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Sequence, Union

import nibabel as nib
import numpy as np
import torch

from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Compose
from monai.utils.enums import CommonKeys
from monai.losses import DiceLoss
from utils.helper import print_gpu_usage, timeit
from utils.transforms import ClickGenerationStrategy, StoppingCriterion

logger = logging.getLogger("interactive_segmentation")
np.seterr(all="raise")

# To debug Nans, slows down code:
# torch.autograd.set_detect_anomaly(True)

# WARNING Code is not tested on batch_size > 1!!!!!!!!!!!!

class Interaction:
    """
    Ignite process_function used to introduce interactions (simulation of clicks) for DeepEdit Training/Evaluation.

    More details about this can be found at:

        Diaz-Pinto et al., MONAI Label: A framework for AI-assisted Interactive
        Labeling of 3D Medical Images. (2022) https://arxiv.org/abs/2203.12362

    Args:
        deepgrow_probability: probability of simulating clicks in an iteration
        transforms: execute additional transformation during every iteration (before train).
            Typically, several Tensor based transforms composed by `Compose`.
        train: True for training mode or False for evaluation mode
        click_probability_key: key to click/interaction probability
        label_names: Dict of label names
        max_interactions: maximum number of interactions per iteration
        TODO: Add more args here!
    """

    def __init__(
        self,
        deepgrow_probability: float,
        transforms: Union[Sequence[Callable], Callable],
        train: bool,
        label_names: Union[None, Dict[str, int]] = None,
        click_probability_key: str = "probability",
        max_interactions: int = 1,
        args=None,
        loss_function=None,
        post_transform=None,
        click_generation_strategy: ClickGenerationStrategy = ClickGenerationStrategy.GLOBAL_CORRECTIVE,
        click_generation_strategy_key: str = "click_generation_strategy",
        stopping_criterion: StoppingCriterion = StoppingCriterion.MAX_ITER,
        iteration_probability: float = 0.5,
        loss_stopping_threshold: float = 0.9,
    ) -> None:
        self.deepgrow_probability = deepgrow_probability
        self.transforms = (
            Compose(transforms) if not isinstance(transforms, Compose) else transforms
        )  # click transforms

        self.train = train
        self.label_names = label_names
        self.click_probability_key = click_probability_key
        self.max_interactions = max_interactions
        self.args = args
        self.loss_function = loss_function
        self.post_transform = post_transform
        self.click_generation_strategy = click_generation_strategy
        self.stopping_criterion = stopping_criterion
        self.iteration_probability = iteration_probability
        self.loss_stopping_threshold = loss_stopping_threshold
        self.click_generation_strategy_key = click_generation_strategy_key
        self.dice_loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

    @timeit
    def __call__(
        self,
        engine: Union[SupervisedTrainer, SupervisedEvaluator],
        batchdata: Dict[str, torch.Tensor],
    ):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        if not self.train:
            # Evaluation does not print epoch / iteration information
            logger.info(
                (
                    f"### Interaction iteration {((engine.state.iteration - 1) % engine.state.epoch_length) + 1}"
                    f"/{engine.state.epoch_length}"
                )
            )
        print_gpu_usage(
            device=engine.state.device,
            used_memory_only=True,
            context="START interaction class",
        )

        # Set up the initial batch data
        in_channels = 1 + len(self.args.labels)
        batchdata_list = decollate_batch(batchdata)
        for i in range(len(batchdata_list)):
            tmp_image = batchdata_list[i][CommonKeys.IMAGE][0 : 0 + 1, ...]
            assert len(tmp_image.shape) == 4
            new_shape = list(tmp_image.shape)
            new_shape[0] = in_channels
            # Set the signal to 0 for all input images
            # image is on channel 0 of e.g. (1,128,128,128) and the signals get appended, so
            # e.g. (3,128,128,128) for two labels
            inputs = torch.zeros(new_shape, device=engine.state.device)
            inputs[0] = batchdata_list[i][CommonKeys.IMAGE][0]
            batchdata_list[i][CommonKeys.IMAGE] = inputs
        batchdata = list_data_collate(batchdata_list)

        iteration = 0
        last_loss = 1
        last_dice = 1
        before_it = time.time()
        while True:
            assert iteration < 1000
            # logger.info(f"{self.stopping_criterion=}")

            if self.stopping_criterion in [
                StoppingCriterion.MAX_ITER,
                StoppingCriterion.MAX_ITER_AND_PROBABILITY,
                StoppingCriterion.MAX_ITER_AND_DICE,
                StoppingCriterion.MAX_ITER_PROBABILITY_AND_DICE,
                StoppingCriterion.DEEPGROW_PROBABILITY,
            ]:
                # Abort if run for max_interactions
                if iteration > self.max_interactions - 1:
                    logger.info("MAX_ITER stop")
                    break
            if self.stopping_criterion in [StoppingCriterion.MAX_ITER_AND_PROBABILITY]:
                # Abort based on the per iteration probability
                if not np.random.choice(
                    [True, False],
                    p=[self.iteration_probability, 1 - self.iteration_probability],
                ):
                    logger.info("PROBABILITY stop")
                    break
            if self.stopping_criterion in [StoppingCriterion.MAX_ITER_AND_DICE]:
                # Abort if dice / loss is good enough
                if last_dice > self.loss_stopping_threshold:
                    logger.info(f"DICE stop, since {last_dice} > {self.loss_stopping_threshold}")
                    break

            if self.stopping_criterion in [
                StoppingCriterion.MAX_ITER_PROBABILITY_AND_DICE,
            ]:
                if np.random.choice([True, False], p=[1 - last_dice, last_dice]):
                    logger.info(f"DICE_PROBABILITY stop, since dice is already {last_dice}")
                    break

            if (
                iteration == 0
                and self.stopping_criterion == StoppingCriterion.DEEPGROW_PROBABILITY
            ):
                # Abort before the first iteration if deepgrow_probability yields False
                if not np.random.choice(
                    [True, False],
                    p=[self.deepgrow_probability, 1 - self.deepgrow_probability],
                ):
                    break

            # NOTE: Image shape e.g. 3x192x192x256, label shape 1x192x192x256
            inputs, labels = engine.prepare_batch(batchdata, device=engine.state.device)
            batchdata[CommonKeys.IMAGE] = inputs
            batchdata[CommonKeys.LABEL] = labels
            # BCHW[D] ?

            if iteration == 0:
                logger.info("inputs.shape is {}".format(inputs.shape))
                # Make sure the signal is empty in the first iteration assertion holds
                assert torch.sum(inputs[:, 1:, ...]) == 0
                logger.info(
                    f"image file name: {batchdata['image_meta_dict']['filename_or_obj']}"
                )
                logger.info(
                    f"label file name: {batchdata['label_meta_dict']['filename_or_obj']}"
                )

                for i in range(len(batchdata["label"][0])):
                    if torch.sum(batchdata["label"][i, 0]) < 0.1:
                        logger.warning(
                            "No valid labels for this sample (probably due to crop)"
                        )

            engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
            engine.network.eval()

            # Forward Pass
            with torch.no_grad():
                if engine.amp:
                    with torch.cuda.amp.autocast():
                        predictions = engine.inferer(inputs, engine.network)
                else:
                    predictions = engine.inferer(inputs, engine.network)

            batchdata[CommonKeys.PRED] = predictions
            
            assert batchdata[CommonKeys.PRED].shape == batchdata[CommonKeys.LABEL].shape, f"{batchdata[CommonKeys.PRED].shape} != {batchdata[CommonKeys.LABEL].shape}"
            # loss = self.loss_function(
            #     batchdata[CommonKeys.PRED], batchdata[CommonKeys.LABEL]
            # )
            # last_loss = loss
            # logger.info(
            #     f"It: {iteration} {self.loss_function.__class__.__name__}: {loss:.4f} Epoch: {engine.state.epoch}"
            # )
            last_dice = self.dice_loss_function(batchdata[CommonKeys.PRED], batchdata[CommonKeys.LABEL])
            logger.info(
                f"It: {iteration} {self.dice_loss_function.__class__.__name__}: {last_dice:.4f} Epoch: {engine.state.epoch}"
            )

            if self.args.save_nifti:
                # tmp_batchdata = {
                #     CommonKeys.PRED: predictions,
                #     CommonKeys.LABEL: batchdata[CommonKeys.LABEL],
                #     "label_names": batchdata["label_names"],
                # }
                tmp_batchdata_list = decollate_batch(tmp_batchdata)
                for i in range(len(tmp_batchdata_list)):
                    tmp_batchdata_list[i] = self.post_transform(tmp_batchdata_list[i])
                tmp_batchdata = list_data_collate(tmp_batchdata_list)

                self.debug_viz(
                    inputs, labels, tmp_batchdata[CommonKeys.PRED], iteration
                )

            # decollate/collate batchdata to execute click transforms
            batchdata_list = decollate_batch(batchdata)
            for i in range(len(batchdata_list)):
                batchdata_list[i][
                    self.click_probability_key
                ] = self.deepgrow_probability
                batchdata_list[i][
                    self.click_generation_strategy_key
                ] = self.click_generation_strategy.value
                batchdata_list[i] = self.transforms(
                    batchdata_list[i]
                )  # Apply click transform

            batchdata = list_data_collate(batchdata_list)

            engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)

            iteration += 1

        logger.info(f"Interaction took {time.time()- before_it:.2f} seconds..")
        # Might be needed for sw_roi_size smaller than 128
        torch.cuda.empty_cache()
        engine.state.batch = batchdata
        return engine._iteration(
            engine, batchdata
        )  # train network with the final iteration cycle

    def debug_viz(self, inputs, labels, preds, j):
        self.save_nifti(
            f"{self.args.data}/guidance_bgg_{j}", inputs[0, 2].cpu().detach().numpy()
        )
        self.save_nifti(
            f"{self.args.data}/guidance_fgg_{j}", inputs[0, 1].cpu().detach().numpy()
        )
        self.save_nifti(f"{self.args.data}/labels", labels[0, 0].cpu().detach().numpy())
        self.save_nifti(f"{self.args.data}/im", inputs[0, 0].cpu().detach().numpy())
        self.save_nifti(
            f"{self.args.data}/pred_{j}", preds[0, 1].cpu().detach().numpy()
        )
        if j == self.max_interactions:
            exit()

    def save_nifti(self, name, im):
        affine = np.eye(4)
        affine[0][0] = -1
        ni_img = nib.Nifti1Image(im, affine=affine)
        ni_img.header.get_xyzt_units()
        ni_img.to_filename(f"{name}.nii.gz")
