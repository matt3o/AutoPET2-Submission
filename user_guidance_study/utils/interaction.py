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

from typing import Callable, Dict, Sequence, Union
import logging
import time
import pprint
import os

import numpy as np
import torch
import nibabel as nib

from monai.data import decollate_batch, list_data_collate
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Compose, AsDiscrete
from monai.utils.enums import CommonKeys
from monai.metrics import compute_dice
from monai.data.meta_tensor import MetaTensor

from utils.helper import print_gpu_usage, get_total_size_of_all_tensors, describe_batch_data, timeit

logger = logging.getLogger("interactive_segmentation")
np.seterr(all='raise')


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
    """

    def __init__(
        self,
        deepgrow_probability: float,
        transforms: Union[Sequence[Callable], Callable],
        train: bool,
        label_names: Union[None, Dict[str, int]] = None,
        click_probability_key: str = "probability",
        max_interactions: int = 1,
        args = None,
    ) -> None:

        self.deepgrow_probability = deepgrow_probability
        self.transforms = Compose(transforms) if not isinstance(transforms, Compose) else transforms # click transforms

        self.train = train
        self.label_names = label_names
        self.click_probability_key = click_probability_key
        self.max_interactions = max_interactions
        self.args = args

    @timeit
    def __call__(self, engine: Union[SupervisedTrainer, SupervisedEvaluator], batchdata: Dict[str, torch.Tensor]):
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")
        guidance_label_overlap = 0.0
        logger.info(f"### Interaction, Epoch {engine.state.epoch}/{engine.state.max_epochs},Iter {engine.state.iteration}/{engine.state.epoch_length}")
        print_gpu_usage(device=engine.state.device, used_memory_only=True, context="START interaction class")
        if np.random.choice([True, False], p=[self.deepgrow_probability, 1 - self.deepgrow_probability]):
            before_it = time.time()
            for j in range(self.max_interactions):
                # logger.info('##### It: {} '.format(j))
                inputs, labels = engine.prepare_batch(batchdata, engine.state.device) # never move directly to device, will loose too much GPU memory

                # inputs = inputs.to(engine.state.device)
                if j == 0:
                    logger.info("inputs.shape is {}".format(inputs.shape))
                # labels = labels.to(engine.state.device)

                engine.fire_event(IterationEvents.INNER_ITERATION_STARTED)
                engine.network.eval()
                #print_gpu_usage(device=engine.state.device, used_memory_only=False, context="Before forward pass")

                # Forward Pass
                with torch.no_grad():
                    if engine.amp:
                        with torch.cuda.amp.autocast():
                            predictions = engine.inferer(inputs, engine.network)
                    else:
                        predictions = engine.inferer(inputs, engine.network)
                
                if self.args.save_nifti or self.args.debug:
                    post_pred = AsDiscrete(argmax=True, to_onehot=2)
                    post_label = AsDiscrete(to_onehot=2)

                    # print(predictions)
                    # print(len(decollate_batch(predictions)))
                    # print(len(decollate_batch(labels)))

                    preds = torch.stack([post_pred(el) for el in decollate_batch(predictions)])
                    gts = torch.stack([post_label(el) for el in decollate_batch(labels)])
                    dice = compute_dice(preds, gts, include_background=True)[0, 1].item()
                    logger.info('It: {} Dice: {:.4f} Epoch: {}'.format(j, dice, engine.state.epoch))

                state = 'train' if self.train else 'eval'

                batchdata.update({CommonKeys.PRED: predictions}) # update predictions of this iteration
                #del predictions

                # decollate/collate batchdata to execute click transforms
                batchdata_list = decollate_batch(batchdata, detach=True)
                

                for i in range(len(batchdata_list)):
                    batchdata_list[i][self.click_probability_key] = self.deepgrow_probability
                    before = time.time()
                    batchdata_list[i] = self.transforms(batchdata_list[i]) # Apply click transform, TODO add patch sized transform
                    # logger.info("self.click_transforms took {:.2f} seconds..".format(time.time()- before))
                    # NOTE: Image size e.g. 3x192x192x256, label size 1x192x192x256

                if j <= 9 and self.args.save_nifti:
                    self.debug_viz(inputs, labels, preds, j)

                batchdata = list_data_collate(batchdata_list)
                # logger.info(describe_batch_data(batchdata, total_size_only=True))
                #del preds, gts, dice
                #del inputs, labels, batchdata_list
                engine.fire_event(IterationEvents.INNER_ITERATION_COMPLETED)
                #print_gpu_usage(device=engine.state.device, used_memory_only=False, context="after It")
            logger.info(f"Interaction took {time.time()- before_it:.2f} seconds..")
        else:
            # zero out input guidance channels
            batchdata_list = decollate_batch(batchdata, detach=True)
            for i in range(1, len(batchdata_list[0][CommonKeys.IMAGE])):
                batchdata_list[0][CommonKeys.IMAGE][i] *= 0
            batchdata = list_data_collate(batchdata_list)
        
        # print_gpu_usage(device=engine.state.device, used_memory_only=True, context="before empty_cache()")
        #torch.cuda.empty_cache()
        #print_gpu_usage(device=engine.state.device, used_memory_only=True, context="END interaction class")
        # first item in batch only
        engine.state.batch = batchdata
        return engine._iteration(engine, batchdata) # train network with the final iteration cycle

    def debug_viz(self, inputs, labels, preds, j):
        self.save_nifti(f'{self.args.data}/guidance_bgg_{j}', inputs[0,2].cpu().detach().numpy())
        self.save_nifti(f'{self.args.data}/guidance_fgg_{j}', inputs[0,1].cpu().detach().numpy())
        self.save_nifti(f'{self.args.data}/labels', labels[0,0].cpu().detach().numpy())
        self.save_nifti(f'{self.args.data}/im', inputs[0,0].cpu().detach().numpy())
        self.save_nifti(f'{self.args.data}/pred_{j}', preds[0,1].cpu().detach().numpy())
        if j == self.max_interactions:
            exit()

    def save_nifti(self, name, im):
        affine = np.eye(4)
        affine[0][0] = -1
        ni_img = nib.Nifti1Image(im, affine=affine)
        ni_img.header.get_xyzt_units()
        ni_img.to_filename(f'{name}.nii.gz')

