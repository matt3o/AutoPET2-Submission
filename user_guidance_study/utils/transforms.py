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

import json
import logging
import random
import warnings
import time
from typing import Dict, Hashable, List, Mapping, Optional, Union, Iterable
import gc

from pynvml import *

import numpy as np
np.seterr(all='raise')
import torch
import pandas as pd
from numpy.typing import ArrayLike

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.transforms import CenterSpatialCropd, Compose, CropForegroundd
from monai.utils import min_version, optional_import
from monai.data.meta_tensor import MetaTensor

import cupy as cp
# Details here: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy
from cupyx.scipy.ndimage import label as label_cp

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

from utils.distance_transform import get_distance_transform, get_choice_from_distance_transform_cp

from utils.helper import print_gpu_usage, print_tensor_gpu_usage, describe, describe_batch_data, timeit
from utils.logger import setup_loggers, get_logger

#logger = logging.getLogger("interactive_segmentation")
#logger.setLevel(logging.INFO)

# Has to be reinitialized for some weird reason here
# Otherwise the logger only works for the click_transforms and never for the pre_transform
setup_loggers()
logger = get_logger()

#distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")
#distance_transform_edt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_edt")

def threshold_foreground(x):
    return x > 0.005

# class GarbageCollectord(MapTransform):
#     def __init__(self, keys: KeysCollection = None):
#         """
#         """
#         super().__init__(keys)

#     def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
#         gc.collect()
#         return data

class NoOpd(MapTransform):
    def __init__(self, keys: KeysCollection = None):
        """
        """
        super().__init__(keys)


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        return data


class DetachTensorsd(MapTransform):
    def __init__(self, keys: KeysCollection = None):
        """
        Detaches all passed tensors.
        """
        super().__init__(keys)


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key].detach()
        # exit(0)
        return d

class CheckTheAmountOfInformationLossByCropd(MapTransform):
    def __init__(self, keys: KeysCollection, roi_size:Iterable, label_names, logger):
        """
        Prints how much information is lost due to the crop.
        """
        super().__init__(keys)
        self.roi_size = roi_size
        self.label_names = label_names
        self.logger = logger

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                new_data = {"label": label.clone(), "image": d["image"].clone()}
                # copy the label and crop it to the desired size
                t = []
                t.append(CropForegroundd(keys=("image", "label"), source_key="image", select_fn=threshold_foreground))
                if self.roi_size is not None:
                    t.append(CenterSpatialCropd(keys="label", roi_size=self.roi_size))
                
                cropped_label = Compose(t)(new_data)["label"]

                # label_num_el = torch.numel(label)
                for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                    # Only count non-background lost labels
                    if key_label != "background":
                        sum_label = torch.sum(label == idx).item()
                        sum_cropped_label = torch.sum(cropped_label == idx).item()
                        # then check how much of the labels is lost
                        lost_pixels = sum_label - sum_cropped_label
                        lost_pixels_ratio = lost_pixels / sum_label * 100
                        self.logger.info(f"{lost_pixels_ratio:.1f} % of labelled pixels of the type {key_label} have been lost when cropping") 
            else: 
                raise UserWarning("This transform only applies to key 'label'")
        return d

class PrintDatad(MapTransform):
    def __init__(self, keys: KeysCollection = None):
        """
        Prints all the information inside data
        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        logger.info(describe_batch_data(d))
        # exit(0)
        return d

class PrintGPUUsaged(MapTransform):
    def __init__(self, device, keys: KeysCollection = None):
        """
        Prints the GPU usage
        """
        super().__init__(keys)
        self.device = device


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        # print_gpu_usage(device=self.device, used_memory_only=True)
        # gc.collect()
        # torch.cuda.empty_cache()
        logger.info(f"Current reserved memory for dataloader: {torch.cuda.memory_reserved(self.device) / (1024**2)} MB")
        # logger.info(torch.cuda.memory_summary())
        # exit(0)
        return d


class InitLoggerd(MapTransform):
    def __init__(self, args, new_logger):
        """ 
        Initialises the logger inside the dataloader thread (if it is a separate thread).

        Has to be reinitialized for some weird reason here, I think this is due to the data transform
        being on an extra thread
        Otherwise the logger only works for the click_transforms and never for the pre_transform
        """
        global logger
        super().__init__(None)
        logger = new_logger
        
        self.loglevel = logging.INFO
        if args.debug:
            self.loglevel = logging.DEBUG

        self.log_file_folder = args.output
        if args.no_log: 
            self.log_file_folder = None

        setup_loggers(self.loglevel, self.log_file_folder)
        logger = get_logger()


    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        global logger
        if logger is None: 
            setup_loggers(self.loglevel, self.log_file_folder)
        logger = get_logger()
        return data


class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(self, keys: KeysCollection, label_names=None, allow_missing_keys: bool = False, device = None):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names
        self.device = device
    
    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Dictionary containing new label numbers
            new_label_names = {}
            label = torch.zeros(d[key].shape, device=self.device)
            # Making sure the range values and number of labels are the same
            for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                if key_label != "background":
                    new_label_names[key_label] = idx
                    label[d[key] == val_label] = idx
                if key_label == "background":
                    new_label_names["background"] = 0

            d["label_names"] = new_label_names
            if isinstance(d[key], MetaTensor):
                d[key].array = label #.to(torch.device("cpu"))
            else:
                d[key] = label #.to(torch.device("cpu"))
        return d


class AddGuidanceSignalDeepEditd(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        guidance_key: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance_key: str = "guidance",
        sigma: int = 3,
        number_intensity_ch: int = 1,
        allow_missing_keys: bool = False,
        disks: bool = False,
        edt: bool = False,
        gdt: bool = False,
        gdt_th: float = 0.1,
        exp_geos: bool = False,
        device = None,
        spacing = None,
        adaptive_sigma = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance_key
        self.sigma = sigma
        self.number_intensity_ch = number_intensity_ch
        self.disks = disks
        self.edt = edt
        self.gdt = gdt
        self.gdt_th = gdt_th
        self.exp_geos = exp_geos
        self.device = device
        self.spacing = spacing
        self.adaptive_sigma = adaptive_sigma

        self.gdt_th = 0 if self.exp_geos else self.gdt_th
        self.gdt = True if self.exp_geos else self.gdt


    def _get_signal(self, image, guidance, key_label):
        dimensions = 3 if len(image.shape) > 3 else 2
        assert type(guidance) == torch.Tensor or type(guidance) == MetaTensor, f"guidance is {type(guidance)}, value {guidance}"
        #guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        #guidance = json.loads(guidance) if isinstance(guidance, str) else guidance

        if self.gdt or self.edt:
            assert self.disks

        if guidance.size()[0]:
            # logger.warning(f"guidance.shape {guidance.shape}")
            first_point_size = guidance[0].numel()
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                assert first_point_size == 4, f" first_point_size is {first_point_size}))"
                signal = torch.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), device=self.device)
            else:
                assert first_point_size == 3, f" first_point_size is {first_point_size}))"
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)

            sshape = signal.shape

            for point in guidance:
                if torch.any(point < 0):
                    continue
                if dimensions == 3:
                    # Making sure points fall inside the image dimension
                    p1 = max(0, min(int(point[-3]), sshape[-3] - 1))
                    p2 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p3 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2, p3] = 1.0
                else:
                    p1 = max(0, min(int(point[-2]), sshape[-2] - 1))
                    p2 = max(0, min(int(point[-1]), sshape[-1] - 1))
                    signal[:, p1, p2] = 1.0

            # Apply a Gaussian filter to the signal
            if torch.max(signal[0]) > 0:
                signal_tensor = signal[0]
                if self.sigma != 0:
                    pt_gaussian = GaussianFilter(len(signal_tensor.shape), sigma=self.sigma)
                    signal_tensor = pt_gaussian(signal_tensor.unsqueeze(0).unsqueeze(0))
                    signal_tensor = signal_tensor.squeeze(0).squeeze(0)

                signal[0] = signal_tensor
                signal[0] = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
                if self.disks:
                    signal[0] = (signal[0] > 0.1) * 1.0 # 0.1 with sigma=1 --> radius = 3, otherwise it is a cube

                    if self.gdt or self.edt or self.adaptive_sigma:
                        raise UserWarning("Code no longer active")
                        fact = 1.0 if (self.gdt or self.exp_geos or self.adaptive_sigma) else 0.0
                        spacing  = self.spacing
                        geos = generalised_geodesic3d(image.unsqueeze(0).to(self.device),
                                                    signal[0].unsqueeze(0).unsqueeze(0).to(self.device),
                                                    spacing,
                                                    10e10,
                                                    fact,
                                                    4)
                        if torch.max(geos.cpu()) > 0:
                            geos = (geos - torch.min(geos)) / (torch.max(geos) - torch.min(geos))
                        vals = geos[0][0].cpu().detach().numpy()

                        if len(vals[vals > 0]) == 0:
                            theta = 0
                        else:
                            theta = np.percentile(vals[vals > 0], self.gdt_th)
                        geos *= ((geos > theta) * 1.0)

                        if self.exp_geos: # Eponentialized Geodesic Distance (MIDeepSeg)
                            geos = 1.0 - torch.exp(-geos)
                        signal[0] = geos[0][0]


            if not (torch.min(signal[0]).item() >= 0 and torch.max(signal[0]).item() <= 1.0):
                raise UserWarning('[WARNING] Bad signal values', torch.min(signal[0]), torch.max(signal[0]))
            if signal is None:
                raise UserWarning("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]), device=self.device)
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]), device=self.device)
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        before = time.time()
        # print_gpu_usage(self.device, used_memory_only=True, context="START AddGuidanceSignalDeepEditd")
        for key in self.key_iterator(d):
            if key == "image":
                image = d[key]
                assert image.is_cuda
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]

                guidance = d[self.guidance_key]
                # e.g. {'spleen': '[[1, 202, 190, 192], [2, 224, 212, 192], [1, 242, 202, 192], [1, 256, 184, 192], [2.0, 258, 198, 118]]', 
                # 'background': '[[257, 0, 98, 118], [1.0, 223, 303, 86]]'}

                for key_label in guidance.keys():
                    # Getting signal based on guidance
                    assert type(guidance[key_label]) == torch.Tensor or type(guidance[key_label]) == MetaTensor, f"guidance[key_label]: {type(guidance[key_label])}\n{guidance[key_label]}"
                    if guidance[key_label] is not None and guidance[key_label].numel():
                        signal = self._get_signal(image, guidance[key_label].to(device=self.device), key_label=key_label)
                    else:
                        signal = self._get_signal(image, torch.Tensor([]).to(device=self.device), key_label=key_label)
                    assert signal.is_cuda
                    assert tmp_image.is_cuda
                    tmp_image = torch.cat([tmp_image, signal], dim=0)
                    # tmp_image = tmp_image.to(torch.device("cpu"))
                    if isinstance(d[key], MetaTensor):
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                logger.debug("AddGuidanceSignalDeepEditd.__call__ took {:.1f} seconds to finish".format(time.time() - before))
                return d
            else:
                raise UserWarning("This transform only applies to image key")
        raise UserWarning("image key has not been been found")


class FindDiscrepancyRegionsDeepEditd(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred_key: key to prediction source.
        discrepancy_key: key to store discrepancies found between label and prediction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred_key: str = "pred",
        discrepancy_key: str = "discrepancy",
        allow_missing_keys: bool = False,
        device = None
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred_key = pred_key
        self.discrepancy_key = discrepancy_key
        self.device = device

    def disparity(self, label, pred):        
        disparity = label - pred
        # +1 means predicted label is not part of the ground truth
        # -1 means predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).to(dtype=torch.float32, device=self.device) #torch.device("cpu")) #.astype(np.float32) # FN
        neg_disparity = (disparity < 0).to(dtype=torch.float32, device=self.device) #torch.device("cpu")) #.astype(np.float32) # FP
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                assert type(d[key]) == torch.Tensor and type(d[self.pred_key]) == torch.Tensor, "{}{}".format(type(d[key]), type(d[self.pred_key]))
                all_discrepancies = {}
                # TODO remove this cuda moval code
                if not d[key].is_cuda:
                    d[key] = d[key].to(device=self.device)
                    label_was_on_cuda = False
                else:
                    label_was_on_cuda = True
                if not d["pred"].is_cuda:
                    d[self.pred_key] = d[self.pred_key].to(device=self.device)
                    pred_was_on_cuda = False
                else:
                    pred_was_on_cuda = True

                # label_names: e.g. [('spleen', 1), ('background', 0)]
                for _, (label_key, label_value) in enumerate(d["label_names"].items()):
                    if label_key != "background":
                        label = torch.clone(d[key].detach())
                        # Label should be represented in 1
                        label[label != label_value] = 0
                        label = (label > 0.5).to(dtype=torch.float32) #.astype(np.float32)

                        # Taking single prediction
                        pred = torch.clone(d[self.pred_key].detach())
                        pred[pred != label_value] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)#.astype(np.float32)
                    else:
                        # TODO look into thos weird conversion - are they necessary?
                        # Taking single label
                        label = torch.clone(d[key].detach())
                        label[label != label_value] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).to(dtype=torch.float32)#.astype(np.float32)
                        # Taking single prediction
                        pred = torch.clone(d[self.pred_key].detach())
                        pred[pred != label_value] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).to(dtype=torch.float32)#.astype(np.float32)
                    all_discrepancies[label_key] = self._apply(label, pred)
                d[self.discrepancy_key] = all_discrepancies
                # Restore previous state of label and pred
                if not label_was_on_cuda:
                    d[key] = d[key].to(device=torch.device("cpu"))
                if not pred_was_on_cuda:
                    d[self.pred_key] = d[self.pred_key].to(device=torch.device("cpu"))
                return d
            else:
                logger.error("This transform only applies to 'label' key")
        raise UserWarning


class AddRandomGuidanceDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance_key: key to guidance source, shape (2, N, # of dim)
        discrepancy_key: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability_key: key to click/interaction probability, shape (1)
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance_key: str = "guidance",
        discrepancy_key: str = "discrepancy",
        probability_key: str = "probability",
        allow_missing_keys: bool = False,
        device=None,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance_key
        self.discrepancy_key = discrepancy_key
        self.probability_key = probability_key
        self._will_interact = None
        self.is_pos = None
        self.is_other = None
        self.default_guidance = None
        self.guidance: Dict[str, List[List[int]]] = {}
        self.device = device

    def randomize(self, data: Dict[Hashable, np.ndarray]):
        probability = data[self.probability_key]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        if not discrepancy.is_cuda:
            discrepancy = discrepancy.to(device=self.device)
        # before = time.time()
        # logger.warning(f"discrepancy.dim: {discrepancy.dim()}")
        distance = get_distance_transform(discrepancy, self.device, verify_correctness=False)
        del discrepancy
        # logger.warning(f"distance.dim(): {distance.dim()}")
        if torch.sum(distance) > 0:
            t = get_choice_from_distance_transform_cp(distance, device=self.device)
            del distance
            return t
        else:
            del distance
            return None

        # # TODO any more GPU stuff possible?
        # if torch.equal(discrepancy, torch.ones_like(discrepancy, device=self.device)):
        #     # special case of the distance, this code shall behave like distance_transform_cdt from scipy
        #     # which means it will return a vector full of -1s in this case
        #     distance = torch.ones_like(discrepancy, device=self.device) * -1
        # else:
        #     with cp.cuda.Device(self.device.index):
        #         discrepancy_cp = cp.asarray(discrepancy.squeeze())
        #         assert len(discrepancy_cp.shape) == 3
        #         distance = torch.as_tensor(distance_transform_edt_cupy(discrepancy_cp), device=self.device)
#        distance = distance.flatten()
#        distance_np = distance.detach().cpu().numpy()
#        probability = np.exp(distance_np) - 1.0
#        idx = np.where(distance_np > 0)[0]
#
#        if torch.sum(distance > 0) > 0:
#            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
#            dst = distance[seed]
#
#            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
#            g[0] = dst[0].item()
#            logger.debug("distance transform in AddRandomGuidance took {:1f} seconds..".format(time.time()- before))
#            return g
#        return None

    def add_guidance(self, guidance, discrepancy, label_names, labels):

        # Positive clicks of the segment in the iteration
        pos_discr = discrepancy[0] # idx 0 is positive discrepancy and idx 1 is negative discrepancy

        # Check the areas that belong to other segments
        # TODO commented since it does nothing..
        # other_discrepancy_areas = {}
        # for _, (key_label, val_label) in enumerate(label_names.items()):
        #     if key_label != "background":
        #         tmp_label = np.copy(labels)
        #         tmp_label[tmp_label != val_label] = 0
        #         tmp_label = (tmp_label > 0.5).astype(np.float32)
        #         other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label) # calculate "area"
        #     else:
        #         tmp_label = np.copy(labels)
        #         tmp_label[tmp_label != val_label] = 1
        #         tmp_label = 1 - tmp_label
        #         other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label) # calculate "area"

        # Add guidance to the current key label
        if torch.sum(pos_discr) > 0:
            tmp_gui = self.find_guidance(pos_discr)
            if tmp_gui is not None:
                # logger.info(f"guidance: {guidance}")
                # logger.info(f"tmp_gui: {torch.Tensor(tmp_gui)}")
                assert guidance.dtype == torch.int32
                guidance = torch.cat((guidance, torch.tensor([tmp_gui], dtype=torch.int32, device=guidance.device)), 0)
                # logger.info(guidance)
#            guidance.append(self.find_guidance(pos_discr)) # sample from positive discrepancy (undersegmentation)
            self.is_pos = True
        return guidance

    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        guidance = d[self.guidance_key]
        discrepancy = d[self.discrepancy_key]
        self.randomize(data)

        if self._will_interact:
            for key_label in d["label_names"].keys():
                tmp_gui = guidance[key_label]
                assert type(tmp_gui) == torch.Tensor or type(tmp_gui) == MetaTensor

#                tmp_gui = tmp_gui.tolist() if isinstance(tmp_gui, np.ndarray) else tmp_gui
#                tmp_gui = json.loads(tmp_gui) if isinstance(tmp_gui, str) else tmp_gui

                if tmp_gui is None:
                    self.guidance[key_label] = torch.tensor([])
                else:
                    # logger.warning(f"tmp_gui: {tmp_gui}")
                    self.guidance[key_label] = tmp_gui[torch.all(tmp_gui >= 0, dim=1).nonzero()].squeeze(1)
                    # logger.warning(f"self.guidance[key_label]: {self.guidance[key_label]}")
                    assert self.guidance[key_label].dim() == 2, f"self.guidance[key_label].shape()  {self.guidance[key_label].shape}"
#                    for row in tmp_gui:
#                        if row.any(-1)
#                        if -1 in row:
#                            continue
#                        else:
#
#                    self.guidance[key_label] = [j for j in tmp_gui if -1 not in j] # Filter guidances with -1 as index

            # Add guidance according to discrepancy
            for key_label in d["label_names"].keys():
                # Add guidance based on discrepancy
                self.guidance[key_label] = self.add_guidance(self.guidance[key_label], discrepancy[key_label], d["label_names"], d["label"])

        if d[self.guidance_key].keys() == self.guidance.keys():
            # d[self.guidance_key] = update_guidance(d[self.guidance_key], self.guidance)
            d[self.guidance_key] = self.guidance
        else:
            raise UserWarning("Can this ever happen?")

        return d


class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation
    """
    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This transform is only for pred key")
        return d


class AddInitialSeedPointMissingLabelsd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.
    Note that the label is of size (C, D, H, W) or (C, H, W)
    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)
    Args:
        guidance_key: key to store guidance.
        sids_key: key that represents lists of valid slice indices for the given label.
        sid_key: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance_key: str = "guidance",
        sids_key: str = "sids",
        sid_key: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
        device = None
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids_key
        self.sid_key = sid_key
        self.sid: Dict[str, int] = dict()
        self.guidance_key = guidance_key
        self.connected_regions = connected_regions
        self.device = device

    def _apply(self, label, sid) -> np.array:
        # sid: single digit, label: array e.g. 1,128,128,128
        assert type(label) == torch.Tensor or type(label) == MetaTensor, "type(label): {} != torch.Tensor or MetaTensor".format(type(label))
        
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][..., sid][None]  # Assume channel is first and depth is last CHWD

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).to(dtype=torch.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        # TODO: 2D code is modified but untested!
        with cp.cuda.Device(self.device.index):
            label_labeled = label_cp(cp.asarray(label))[0]
            blobs_labels = torch.as_tensor(label_labeled, device=self.device) if dims == 2 else label
#            blobs_labels = torch.from_numpy(measure.label(label.to(dtype=torch.int32).cpu(), background=0)).to(device=self.device) if dims == 2 else label

        label_guidance = []
        # If the label is not present in this slice
        if torch.max(blobs_labels) <= 0:
            label_guidance.append(self.default_guidance)
        else:
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).to(dtype=torch.float32)
                    if torch.sum(label) == 0:
                        label_guidance.append(self.default_guidance)
                        continue

                distance = get_distance_transform(label, self.device, verify_correctness=False)
                g = get_choice_from_distance_transform_cp(distance, device=self.device)
                if dimensions == 2 or dims == 3:
                    label_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    label_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD
        return torch.tensor(label_guidance, dtype=torch.int32, device=self.device)#torch.device("cpu"))

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key).get(key_label) if d.get(self.sids_key) is not None else None
        sid = d.get(self.sid_key).get(key_label) if d.get(self.sid_key) is not None else None
        if sids is not None and sids.size:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.warning(f"No slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d[self.sids_key].keys():
                    # Randomize: Select a random slice
                    self._randomize(d, key_label)
                    # Generate guidance base on selected slice
                    tmp_label = torch.clone(d[key].detach())
                    assert tmp_label.is_cuda
                    # Taking one label to create the guidance       
                    if key_label != "background":
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                    else:
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                        tmp_label = 1 - tmp_label

                    label_guidances[key_label] = self._apply(tmp_label, self.sid.get(key_label))
                    # logger.warning(f"label_guidances[key_label] is {label_guidances[key_label]}")

                if self.guidance_key in d.keys():
                    #d[self.guidance_key] = update_guidance(d[self.guidance_key], label_guidances)
                    d[self.guidance_key] = label_guidances
                else:
                    d[self.guidance_key] = label_guidances # Initialize Guidance Dict
                del d[self.sids_key]
                return d
            else:
                raise UserWarning("This transform only applies to label key")
        raise UserWarning("No input to AddInitialSeedPointMissingLabelsd")


class FindAllValidSlicesMissingLabelsd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.
    Args:
        sids_key: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids_key="sids", allow_missing_keys: bool = False, device=None):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids_key
        self.device = device

    def _apply(self, label, d):
        assert type(label) == torch.Tensor or type(label) == MetaTensor
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):      # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    # Append the item instead of the 1-d Tensor value
                    l_ids.append(sid)
            # If there are not slices with the label
            if not len(l_ids):
                l_ids = [-1] * 10
            sids[key_label] = np.asarray(l_ids)
        return sids

    # @torch.no_grad()
    @timeit
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label = d[key]
                if label.shape[0] != 1:
                    raise ValueError("Only supports single channel labels!")

                if len(label.shape) != 4:  # only for 3D
                    raise ValueError("Only supports label with shape CHWD!")

                sids = self._apply(label, d)
                if sids is not None and len(sids.keys()):
                    d[self.sids_key] = sids
                return d
            else:
                raise UserWarning("This transform only applies to label key")
        raise UserWarning("No input to FindAllValidSlicesMissingLabelsd")
