import logging
from typing import Dict, Hashable, List, Mapping, Optional, Union

import torch
import numpy as np

np.seterr(all="raise")

import cupy as cp

# Details here: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt
from cucim.core.operations.morphology import (
    distance_transform_edt as distance_transform_edt_cupy,
)

from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from numpy.typing import ArrayLike

from utils.helper import (
    print_gpu_usage,
    print_tensor_gpu_usage,
    describe,
    describe_batch_data,
    timeit,
)

logger = logging.getLogger("interactive_segmentation")

"""
CUDA enabled distance transforms using cupy
"""


def find_discrepancy(
    vec1: ArrayLike,
    vec2: ArrayLike,
    context_vector: ArrayLike,
    atol: float = 0.001,
    raise_warning: bool = True,
):
    if not np.allclose(vec1, vec2):
        logger.error("find_discrepancy() found something")
        idxs = np.where(np.isclose(vec1, vec2) == False)
        assert len(idxs) > 0 and idxs[0].size > 0
        for i in range(0, min(5, idxs[0].size)):
            position = []
            for j in range(0, len(vec1.shape)):
                position.append(idxs[j][i])
            position = tuple(position)
            logger.error("{} \n".format(position))
            logger.error(
                "Item at position: {} which has value: {} \nvec1: {} , vec2: {}".format(
                    position,
                    context_vector.squeeze()[position],
                    vec1[position],
                    vec2[position],
                )
            )
        if raise_warning:
            raise UserWarning(
                "find_discrepancy has found discrepancies! Please fix your code.."
            )


def get_distance_transform(
    tensor: torch.Tensor, device: torch.device = None, verify_correctness=False
) -> torch.Tensor:
    # The distance transform provides a metric or measure of the separation of points in the image.
    # This function calculates the distance between each pixel that is set to off (0) and
    # the nearest nonzero pixel for binary images
    # http://matlab.izmiran.ru/help/toolbox/images/morph14.html
    dimension = tensor.dim()
    if verify_correctness:
        distance_np = distance_transform_edt(tensor.cpu().numpy())
    # Check is necessary since the edt transform only accepts certain dimensions
    if dimension == 4:
        tensor = tensor.squeeze(0)
    assert (
        len(tensor.shape) == 3 and tensor.is_cuda
    ), "tensor.shape: {}, tensor.is_cuda: {}".format(tensor.shape, tensor.is_cuda)
    special_case = False
    if torch.equal(tensor, torch.ones_like(tensor, device=device)):
        # special case of the distance, this code shall behave like distance_transform_cdt from scipy
        # which means it will return a vector full of -1s in this case
        # Otherwise there is a corner case where if all items in label are 1, the distance will become inf..
        distance = torch.ones_like(tensor, device=device)  # * -1
        special_case = True
    else:
        with cp.cuda.Device(device.index):
            tensor_cp = cp.asarray(tensor)
            distance = torch.as_tensor(
                distance_transform_edt_cupy(tensor_cp), device=device
            )

    if verify_correctness and not special_case:
        distance = torch.as_tensor(
            distance_transform_edt_cupy(tensor_cp), device=device
        )
        find_discrepancy(distance_np, distance.cpu().numpy(), tensor)

    if dimension == 4:
        distance = distance.unsqueeze(0)
    assert distance.dim() == dimension
    return distance


def get_choice_from_distance_transform_cp(
    distance: torch.Tensor, device: torch.device, max_threshold: int = None
) -> List[int | List[int]] | None:
    if torch.sum(distance) <= 0:
        # dont raise, just return an empty result
        return None

    with cp.cuda.Device(device.index):
        if max_threshold is None:
            # divide by the maximum number of elements in a volume
            max_threshold = int(cp.floor(cp.log(cp.finfo(cp.float32).max))) / (
                800 * 800 * 800
            )

        # Clip the distance transform to avoid overflows and negative probabilities
        clipped_distance = distance.clip(min=0, max=max_threshold)
        distance_cp = cp.asarray(clipped_distance)

        g = get_choice_from_tensor(distance_cp, device, max_threshold, size=1)

    return g


def get_choice_from_tensor(
    t: torch.Tensor | cp.ndarray,
    device: torch.device,
    max_threshold: int = None,
    size=1,
) -> List[int | List[int]] | None:
    with cp.cuda.Device(device.index):
        if not isinstance(t, cp.ndarray):
            t_cp = cp.asarray(t)
        else:
            t_cp = t

        if cp.sum(t_cp) <= 0:
            # No valid distance has been found. Dont raise, just empty return
            return None

        flattened_t_cp = t_cp.flatten()

        probability = cp.exp(flattened_t_cp) - 1.0
        idx = cp.where(flattened_t_cp > 0)[0]
        probabilities = probability[idx] / cp.sum(probability[idx])
        assert idx.shape == probabilities.shape
        assert cp.all(cp.greater_equal(probabilities, 0))

        seed = cp.random.choice(a=idx, size=size, p=probabilities)
        dst = flattened_t_cp[seed.item()]

        g = cp.asarray(cp.unravel_index(seed, t_cp.shape)).transpose().tolist()[0]
        g[0] = dst.item()
    assert len(g) == len(
        t_cp.shape
    ), f"g has wrong dimensions! {len(g)} != {len(t_cp.shape)}"
    return g
