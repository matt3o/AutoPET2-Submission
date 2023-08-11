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

from typing import Callable, Sequence, Union, Dict, Any
import logging
import shutil
import json
import pathlib
import copy
import torch
import numpy as np
import nibabel as nib

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    Fetch2DSliced,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monailabel.transform.post import Restored
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToNumpyd,
    SqueezeDimd,
    MapTransform,
    Compose
)
from monai.transforms.transform import MapTransform
from monai.data import decollate_batch

from monailabel.interfaces.utils.transform import run_transforms
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask, CallBackTypes
from sw_interactive_segmentation.api import (
    get_pre_transforms, 
    get_post_transforms,
    get_inferers,
    get_pre_transforms_val_as_list_monailabel,
)
from sw_interactive_segmentation.utils.helper import AttributeDict
from sw_interactive_segmentation.utils.transforms import AddGuidanceSignal, PrintDatad

from monai.utils import set_determinism
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class SWInteractiveSegmentationInfer(BasicInferTask):

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        label_names=None,
        dimension=3,
        description="",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.label_names = label_names

        self.args = AttributeDict()
        self.args.no_log = True
        self.args.output = None
        self.args.output_dir = None
        self.args.dataset = "AutoPET"
        self.args.train_crop_size = (128,128,128)
        self.args.val_crop_size = None
        self.args.inferer = "SlidingWindowInferer"
        self.args.sw_roi_size = (128,128,128)
        self.args.train_sw_batch_size = 8
        self.args.val_sw_batch_size = 24
        self.args.debug = False
        self.args.path = '/projects/mhadlich_segmentation/data/monailabel'
        set_determinism(42)
        self.model_state_dict = "net"
        self.load_strict = True


    def pre_transforms(self, data=None) -> Sequence[Callable]:
        print("#########################################")

        data['label_dict'] = self.label_names
        data['label_names'] = self.label_names
        device = data.get("device") if data else None
        t = []
        t_val_1, t_val_2 = get_pre_transforms_val_as_list_monailabel(self.label_names, device, self.args, input_keys=["image"])
        t.extend(t_val_1)
        self.add_cache_transform(t, data)
        t.extend(t_val_2)
        #t_val = []
        #t_val.append(NoOpd())
        #t_val.append(LoadImaged(keys="image", reader="ITKReader"))

        #t_val.append(NoOpd())
        return t

    def inferer(self, data=None) -> Inferer:
        _, val_inferer = get_inferers(
            inferer=self.args.inferer,
            sw_roi_size=self.args.sw_roi_size,
            train_crop_size=self.args.train_crop_size,
            val_crop_size=self.args.val_crop_size,
            train_sw_batch_size=self.args.train_sw_batch_size,
            val_sw_batch_size=self.args.val_sw_batch_size,
            cache_roi_weight_map=False,
        )
        return val_inferer

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        device = data.get("device") if data else None
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            Restored(keys="pred", ref_image="image"),
            EnsureTyped(keys="pred", device="cpu" if data else None, dtype=torch.uint8),
        ]

    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """
        logger.info("#################")
        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")
        
        data["path"] = self.args.path
        network = self._get_network(device, data)
        if network:
            inputs = data[self.input_key]
            inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
            inputs = inputs[None] if convert_to_batch else inputs
            inputs = inputs.to(torch.device(device))

            with torch.no_grad():
                outputs = inferer(inputs, network)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            if convert_to_batch:
                if isinstance(outputs, dict):
                    outputs_d = decollate_batch(outputs)
                    outputs = outputs_d[0]
                else:
                    outputs = outputs[0]

            data[self.output_label_key] = outputs
        else:
            # consider them as callable transforms
            data = run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
        
        return data

    def __call__(self, request, callbacks= None):
        if callbacks is None:
            callbacks = {}
        callbacks[CallBackTypes.POST_TRANSFORMS] = post_callback
        
        return super().__call__(request, callbacks)

    def run_invert_transforms(self, data: Dict[str, Any], pre_transforms, names):
        if names is None:
            return data

        pre_names = dict()
        transforms = []
        for t in reversed(pre_transforms):
            if hasattr(t, "inverse"):
                pre_names[t.__class__.__name__] = t
                transforms.append(t)

        # Run only selected/given
        if len(names) > 0:
            transforms = [pre_transforms[n if isinstance(n, str) else n.__name__] for n in names]


        d = run_transforms(data, transforms, inverse=True, log_prefix="INV")
        d = copy.deepcopy(dict(data))
        d[self.input_key] = data[self.output_label_key]
        d = run_transforms(d, transforms, inverse=True, log_prefix="INV")
        data[self.output_label_key] = d[self.input_key]
        return data


def post_callback(data): 
    path = '/projects/mhadlich_segmentation/data/monailabel'
    
    #for k,v in data.items():
    #    print(f"k: {k}, v: {v}")
    image_name = Path(os.path.basename(data["image_path"]))
    true_image_name = image_name.name.removesuffix(''.join(image_name.suffixes))
    image_folder = Path(data["image_path"]).parent
    print(f"{true_image_name=}")
    print(f"{image_folder}")
    labels_folder = os.path.join(image_folder, "labels", "final") 

    if not os.path.exists(labels_folder):
        print(f"###### Creating {labels_folder}")
        pathlib.Path(labels_folder).mkdir(parents=True)

    # Save the clicks
    clicks_per_label = {}
    for key in data['label_dict'].keys():
        clicks_per_label[key] = data[key]
        assert isinstance(data[key], list)
    click_file_path = os.path.join(labels_folder, f"{true_image_name}_clicks.json")
    logger.info(f"Now dumping dict: {clicks_per_label} to file {click_file_path} ...")
    with open(click_file_path, "w") as clicks_file:
        json.dump(clicks_per_label, clicks_file)

    # Save debug NIFTI, not fully working since the inverse transform of the image is not avaible
    if False:
        logger.info("SAVING NIFTI")
        inputs = data["image"]
        pred = data["pred"]
        logger.info(f"inputs.shape is {inputs.shape}")
        logger.info(f"sum of fgg is {torch.sum(inputs[1])}")
        logger.info(f"sum of bgg is {torch.sum(inputs[2])}")
        logger.info(f"Image path is {data['image_path']}, copying file")
        shutil.copyfile(data['image_path'], f"{path}/im.nii.gz")
        #save_nifti(f"{path}/im", inputs[0].cpu().detach().numpy())
        save_nifti(
            f"{path}/guidance_fgg", inputs[1].cpu().detach().numpy()
        )
        save_nifti(
            f"{path}/guidance_bgg", inputs[2].cpu().detach().numpy()
        )
        logger.info(f"pred.shape is {pred.shape}")
        save_nifti(
            f"{path}/pred", pred.cpu().detach().numpy()
        )
    return data

def save_nifti(name, im):
    affine = np.eye(4)
    affine[0][0] = -1
    ni_img = nib.Nifti1Image(im, affine=affine)
    ni_img.header.get_xyzt_units()
    ni_img.to_filename(f"{name}.nii.gz")

class NoOpd(MapTransform):
    def __init__(self, keys= None):
        """
        A transform which does nothing
        """
        super().__init__(keys)

    def __call__(
        self, data
        ):
        #print(data["image"])
        try:
            print(data["image"])
            print(data["image_path"])
        except AttributeError:
            pass
        print(type(data["image"]))
        return data




