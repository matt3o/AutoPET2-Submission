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
from typing import Dict, Hashable, List, Mapping, Optional

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data import MetaTensor
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import MapTransform, Randomizable, Transform
from monai.utils import min_version, optional_import

from FastGeodis import generalised_geodesic3d

# Add new click to the guidance signal
def update_guidance(orig, updated):
    assert orig.keys() == updated.keys()
    for k in orig.keys():
        v_new = updated[k]
        v_old = eval(orig[k]) # str->list
        for p in v_new:
            if p not in v_old:
                v_old.append(p)
        v_old = [el for el in v_old if np.min(el) >= 0]
        orig[k] = str(v_old)
    return orig

measure, _ = optional_import("skimage.measure", "0.14.2", min_version)

logger = logging.getLogger(__name__)


distance_transform_cdt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_cdt")

class NormalizeLabelsInDatasetd(MapTransform):
    def __init__(self, keys: KeysCollection, label_names=None, allow_missing_keys: bool = False):
        """
        Normalize label values according to label names dictionary

        Args:
            keys: The ``keys`` parameter will be used to get and set the actual data item to transform
            label_names: all label names
        """
        super().__init__(keys, allow_missing_keys)

        self.label_names = label_names

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            # Dictionary containing new label numbers
            new_label_names = {}
            label = np.zeros(d[key].shape)
            # Making sure the range values and number of labels are the same
            for idx, (key_label, val_label) in enumerate(self.label_names.items(), start=1):
                if key_label != "background":
                    new_label_names[key_label] = idx
                    label[d[key] == val_label] = idx
                if key_label == "background":
                    new_label_names["background"] = 0

            d["label_names"] = new_label_names
            if isinstance(d[key], MetaTensor):
                d[key].array = label
            else:
                d[key] = label
        return d


class AddGuidanceSignalDeepEditd(MapTransform):
    """
    Add Guidance signal for input image. Multilabel DeepEdit

    Based on the "guidance" points, apply Gaussian to them and add them as new channel for input image.

    Args:
        guidance: key to store guidance.
        sigma: standard deviation for Gaussian kernel.
        number_intensity_ch: channel index.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
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
        adaptive_sigma = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance = guidance
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


    def _get_signal(self, image, guidance, key_label='spleen'):
        dimensions = 3 if len(image.shape) > 3 else 2
        guidance = guidance.tolist() if isinstance(guidance, np.ndarray) else guidance
        guidance = json.loads(guidance) if isinstance(guidance, str) else guidance

        if self.gdt or self.edt:
            assert self.disks

        if len(guidance):
            if dimensions == 3:
                # Assume channel is first and depth is last CHWD
                signal = torch.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]))
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]))

            sshape = signal.shape

            for point in guidance:
                if torch.any(torch.asarray(point) < 0):
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


            if not (torch.min(signal[0]) >= 0 and torch.max(signal[0] <= 1.0)):
                print('[WARNING] Bad signal values', torch.min(signal[0]), torch.max(signal[0]))
            if signal is None:
                print("[ERROR] Signal is None")
            return signal
        else:
            if dimensions == 3:
                signal = torch.zeros((1, image.shape[-3], image.shape[-2], image.shape[-1]))
            else:
                signal = torch.zeros((1, image.shape[-2], image.shape[-1]))
            if signal is None:
                print("[ERROR] Signal is None")
            return signal

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "image":
                image = d[key]
                tmp_image = image[0 : 0 + self.number_intensity_ch, ...]
                guidance = d[self.guidance]

                for key_label in guidance.keys():
                    # Getting signal based on guidance
                    if guidance[key_label] is not None and len(guidance[key_label]):
                        signal = self._get_signal(image, guidance[key_label], key_label=key_label)
                    else:
                        signal = self._get_signal(image, [])
                    tmp_image = torch.cat([tmp_image.cpu(), signal], dim=0)
                    if isinstance(d[key], MetaTensor):
                        d[key].array = tmp_image
                    else:
                        d[key] = tmp_image
                return d
            else:
                print("This transform only applies to image key")

        return d

class FindDiscrepancyRegionsDeepEditd(MapTransform):
    """
    Find discrepancy between prediction and actual during click interactions during training.

    Args:
        pred: key to prediction source.
        discrepancy: key to store discrepancies found between label and prediction.
    """

    def __init__(
        self,
        keys: KeysCollection,
        pred: str = "pred",
        discrepancy: str = "discrepancy",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.pred = pred
        self.discrepancy = discrepancy

    @staticmethod
    def disparity(label, pred):
        disparity = label - pred
        # +1 means predicted label is not part of the ground truth
        # -1 means predicted label missed that region of the ground truth
        pos_disparity = (disparity > 0).astype(np.float32) # FN
        neg_disparity = (disparity < 0).astype(np.float32) # FP
        return [pos_disparity, neg_disparity]

    def _apply(self, label, pred):
        return self.disparity(label, pred)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                all_discrepancies = {}
                for _, (key_label, val_label) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        # Taking single label
                        label = np.copy(d[key])

                        # Label should be represented in 1
                        label[label != val_label] = 0
                        label = (label > 0.5).astype(np.float32)

                        # Taking single prediction
                        pred = np.copy(d[self.pred])
                        pred[pred != val_label] = 0
                        # Prediction should be represented in one
                        pred = (pred > 0.5).astype(np.float32)
                    else:
                        # Taking single label
                        label = np.copy(d[key])
                        label[label != val_label] = 1
                        label = 1 - label
                        # Label should be represented in 1
                        label = (label > 0.5).astype(np.float32)
                        # Taking single prediction
                        pred = np.copy(d[self.pred])
                        pred[pred != val_label] = 1
                        pred = 1 - pred
                        # Prediction should be represented in one
                        pred = (pred > 0.5).astype(np.float32)
                    all_discrepancies[key_label] = self._apply(label, pred)
                d[self.discrepancy] = all_discrepancies

                return d
            else:
                print("This transform only applies to 'label' key")
        return d # should never end up here... (dead code)


class AddRandomGuidanceDeepEditd(Randomizable, MapTransform):
    """
    Add random guidance based on discrepancies that were found between label and prediction.

    Args:
        guidance: key to guidance source, shape (2, N, # of dim)
        discrepancy: key to discrepancy map between label and prediction shape (2, C, H, W, D) or (2, C, H, W)
        probability: key to click/interaction probability, shape (1)
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        discrepancy: str = "discrepancy",
        probability: str = "probability",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.guidance_key = guidance
        self.discrepancy = discrepancy
        self.probability = probability
        self._will_interact = None
        self.is_pos = None
        self.is_other = None
        self.default_guidance = None
        self.guidance: Dict[str, List[List[int]]] = {}

    def randomize(self, data=None):
        probability = data[self.probability]
        self._will_interact = self.R.choice([True, False], p=[probability, 1.0 - probability])

    def find_guidance(self, discrepancy):
        distance = distance_transform_cdt(discrepancy).flatten()
        probability = np.exp(distance.flatten()) - 1.0
        idx = np.where(discrepancy.flatten() > 0)[0]

        if np.sum(discrepancy > 0) > 0:
            seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
            dst = distance[seed]

            g = np.asarray(np.unravel_index(seed, discrepancy.shape)).transpose().tolist()[0]
            g[0] = dst[0]
            return g
        return None

    def add_guidance(self, guidance, discrepancy, label_names, labels):

        # Positive clicks of the segment in the iteration
        pos_discr = discrepancy[0]  # idx 0 is positive discrepancy and idx 1 is negative discrepancy

        # Check the areas that belong to other segments
        other_discrepancy_areas = {}
        for _, (key_label, val_label) in enumerate(label_names.items()):
            if key_label != "background":
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 0
                tmp_label = (tmp_label > 0.5).astype(np.float32)
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label) # calculate "area"
            else:
                tmp_label = np.copy(labels)
                tmp_label[tmp_label != val_label] = 1
                tmp_label = 1 - tmp_label
                other_discrepancy_areas[key_label] = np.sum(discrepancy[1] * tmp_label) # calculate "area"

        # Add guidance to the current key label
        if np.sum(pos_discr) > 0:
            guidance.append(self.find_guidance(pos_discr)) # sample from positive discrepancy (undersegmentation)
            self.is_pos = True



    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)

        guidance = d[self.guidance_key]
        discrepancy = d[self.discrepancy]
        self.randomize(data)

        if self._will_interact:
            # Convert all guidance to lists so new guidance can be easily appended
            for key_label in d["label_names"].keys():
                tmp_gui = guidance[key_label]
                tmp_gui = tmp_gui.tolist() if isinstance(tmp_gui, np.ndarray) else tmp_gui
                tmp_gui = json.loads(tmp_gui) if isinstance(tmp_gui, str) else tmp_gui

                if tmp_gui is None:
                    self.guidance[key_label] = []
                else:
                    self.guidance[key_label] = [j for j in tmp_gui if -1 not in j] # Filter guidances with -1 as index

            # Add guidance according to discrepancy
            for key_label in d["label_names"].keys():
                # Add guidance based on discrepancy
                self.add_guidance(self.guidance[key_label], discrepancy[key_label], d["label_names"], d["label"])

        if d[self.guidance_key].keys() == self.guidance.keys():
            d[self.guidance_key] = update_guidance(d[self.guidance_key], self.guidance)


        return d

class SplitPredsLabeld(MapTransform):
    """
    Split preds and labels for individual evaluation

    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "pred":
                for idx, (key_label, _) in enumerate(d["label_names"].items()):
                    if key_label != "background":
                        d[f"pred_{key_label}"] = d[key][idx + 1, ...][None]
                        d[f"label_{key_label}"] = d["label"][idx + 1, ...][None]
            elif key != "pred":
                logger.info("This is only for pred key")
        return d


class AddInitialSeedPointMissingLabelsd(Randomizable, MapTransform):
    """
    Add random guidance as initial seed point for a given label.
    Note that the label is of size (C, D, H, W) or (C, H, W)
    The guidance is of size (2, N, # of dims) where N is number of guidance added.
    # of dims = 4 when C, D, H, W; # of dims = 3 when (C, H, W)
    Args:
        guidance: key to store guidance.
        sids: key that represents lists of valid slice indices for the given label.
        sid: key that represents the slice to add initial seed point.  If not present, random sid will be chosen.
        connected_regions: maximum connected regions to use for adding initial points.
    """

    def __init__(
        self,
        keys: KeysCollection,
        guidance: str = "guidance",
        sids: str = "sids",
        sid: str = "sid",
        connected_regions: int = 5,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.sids_key = sids
        self.sid_key = sid
        self.sid: Dict[str, int] = dict()
        self.guidance = guidance
        self.connected_regions = connected_regions

    def _apply(self, label, sid):
        dimensions = 3 if len(label.shape) > 3 else 2
        self.default_guidance = [-1] * (dimensions + 1)

        dims = dimensions
        if sid is not None and dimensions == 3:
            dims = 2
            label = label[0][..., sid][np.newaxis]  # Assume channel is first and depth is last CHWD

        # THERE MAY BE MULTIPLE BLOBS FOR SINGLE LABEL IN THE SELECTED SLICE
        label = (label > 0.5).astype(np.float32)
        # measure.label: Label connected regions of an integer array - Two pixels are connected
        # when they are neighbors and have the same value
        blobs_labels = measure.label(label.astype(int), background=0) if dims == 2 else label

        label_guidance = []
        # If there are is presence of that label in this slice
        if np.max(blobs_labels) <= 0:
            label_guidance.append(self.default_guidance)
        else:
            for ridx in range(1, 2 if dims == 3 else self.connected_regions + 1):
                if dims == 2:
                    label = (blobs_labels == ridx).astype(np.float32)
                    if np.sum(label) == 0:
                        label_guidance.append(self.default_guidance)
                        continue

                # The distance transform provides a metric or measure of the separation of points in the image.
                # This function calculates the distance between each pixel that is set to off (0) and
                # the nearest nonzero pixel for binary images
                # http://matlab.izmiran.ru/help/toolbox/images/morph14.html
                distance = distance_transform_cdt(label).flatten()
                probability = np.exp(distance) - 1.0

                idx = np.where(label.flatten() > 0)[0]
                seed = self.R.choice(idx, size=1, p=probability[idx] / np.sum(probability[idx]))
                dst = distance[seed]

                g = np.asarray(np.unravel_index(seed, label.shape)).transpose().tolist()[0]
                g[0] = dst[0]  # for debug
                if dimensions == 2 or dims == 3:
                    label_guidance.append(g)
                else:
                    # Clicks are created using this convention Channel Height Width Depth (CHWD)
                    label_guidance.append([g[0], g[-2], g[-1], sid])  # Assume channel is first and depth is last CHWD

        return np.asarray(label_guidance)

    def _randomize(self, d, key_label):
        sids = d.get(self.sids_key).get(key_label) if d.get(self.sids_key) is not None else None
        sid = d.get(self.sid_key).get(key_label) if d.get(self.sid_key) is not None else None
        if sids is not None and sids:
            if sid is None or sid not in sids:
                sid = self.R.choice(sids, replace=False)
        else:
            logger.info(f"Not slice IDs for label: {key_label}")
            sid = None
        self.sid[key_label] = sid

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d: Dict = dict(data)
        for key in self.key_iterator(d):
            if key == "label":
                label_guidances = {}
                for key_label in d["sids"].keys():
                    # Randomize: Select a random slice
                    self._randomize(d, key_label)
                    # Generate guidance base on selected slice
                    tmp_label = np.copy(d[key])
                    # Taking one label to create the guidance
                    if key_label != "background":
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 0
                    else:
                        tmp_label[tmp_label != float(d["label_names"][key_label])] = 1
                        tmp_label = 1 - tmp_label
                    label_guidances[key_label] = json.dumps(
                        self._apply(tmp_label, self.sid.get(key_label)).astype(int).tolist()
                    )

                if self.guidance in d.keys():
                    d[self.guidance] = update_guidance(d[self.guidance], label_guidances)
                else:
                    d[self.guidance] = label_guidances # Initialize Guidance Dict

                return d
            else:
                print("This transform only applies to label key")

        return d


class FindAllValidSlicesMissingLabelsd(MapTransform):
    """
    Find/List all valid slices in the labels.
    Label is assumed to be a 4D Volume with shape CHWD, where C=1.
    Args:
        sids: key to store slices indices having valid label map.
    """

    def __init__(self, keys: KeysCollection, sids="sids", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.sids = sids

    def _apply(self, label, d):
        sids = {}
        for key_label in d["label_names"].keys():
            l_ids = []
            for sid in range(label.shape[-1]):  # Assume channel is first and depth is last CHWD
                if d["label_names"][key_label] in label[0][..., sid]:
                    l_ids.append(sid)
            # If there are not slices with the label
            if l_ids == []:
                l_ids = [-1] * 10
            sids[key_label] = l_ids
        return sids

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
                    d[self.sids] = sids
                return d
            else:
                print("This transform only applies to label key")
        return d
