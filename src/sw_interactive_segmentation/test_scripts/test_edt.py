from __future__ import annotations

import logging

import cupy as cp
import numpy as np
import torch
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy

# from FastGeodis import generalised_geodesic3d
from scipy.ndimage.morphology import distance_transform_cdt as distance_transform_cdt_scipy
from scipy.ndimage.morphology import distance_transform_edt as distance_transform_edt_scipy

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

import time


def main():
    vector_t = torch.Tensor(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
        ]
    ).to(device=torch.device("cuda:3"))
    # vector_t = torch.load("discrepancy.pt").squeeze()
    #    vector_t = torch.rand((128,128,128))
    vector_t = torch.ones((200, 200, 200)).to(device=torch.device("cuda:3"))
    vector_t[0][0][0] = 0
    logger.info(vector_t.shape)

    iterations = 1

    before = time.time()
    vector = vector_t.clone().cpu().numpy()
    for i in range(iterations):
        distance = distance_transform_cdt_scipy(vector)
    elapsed_time = time.time() - before
    logger.info(
        "distance_transform_cdt_scipy: {} runs took {:f} seconds, which means {:f} seconds per run".format(
            iterations, elapsed_time, elapsed_time / iterations
        )
    )
    print(distance)

    before = time.time()
    vector = vector_t.clone().cpu().numpy()
    for i in range(iterations):
        distance = distance_transform_edt_scipy(vector)
    elapsed_time = time.time() - before
    logger.info(
        "distance_transform_edt_scipy: {} runs took {:f} seconds, which means {:f} seconds per run".format(
            iterations, elapsed_time, elapsed_time / iterations
        )
    )
    print(distance)

    before = time.time()
    vector_cp = cp.asarray(vector_t)
    for i in range(iterations):
        distance_cp = distance_transform_edt_cupy(vector_cp)
    elapsed_time = time.time() - before
    logger.info(
        "distance_transform_edt_cupy: {} runs took {:f} seconds, which means {:f} seconds per run.".format(
            iterations, elapsed_time, elapsed_time / iterations
        )
    )
    print(torch.Tensor(distance_cp))
    # assert np.allclose(distance, distance_cp, atol=0.001)
    find_discrepancy(distance, torch.Tensor(distance_cp), vector_t, 0.001)

    # DISABLED since it yields highly different results than distance_transform_edt_scipy..
    # Can still be run, will also yield the areas where the vectors differ but that did not really help imo
    # d_edt_gpu = generalised_geodesic3d(vector_t,
    #                                     vector_t,
    #                                     [1.0, 1.0, 1.0],
    #                                     10e10,
    #                                     0.0,
    #                                     2)
    # distance = distance.squeeze()
    # d_edt_gpu = d_edt_gpu.squeeze()
    # logger.info("d_edt_gpu:\n {}".format(d_edt_gpu))


def find_discrepancy(vec1, vec2, context_vector, atol=0.001):
    if not np.allclose(vec1, vec2):
        # logger.error(np.logical_not(np.isclose(vec1, vec2)))
        idxs = np.where(np.isclose(vec1, vec2) == False)
        assert len(idxs) > 0
        for i in range(0, min(5, idxs[0].size)):
            position = []
            for j in range(0, len(vec1.shape)):
                position.append(idxs[j][i])
            position = tuple(position)
            logger.info("{} \n".format(position))
            logger.info(
                "Item at position: {} which has value: {} \nvec1: {} , vec2: {}".format(
                    position,
                    context_vector.squeeze()[position],
                    vec1[position],
                    vec2[position],
                )
            )
            # logger.info("Context array: {}".format(context_vector.squeeze()[max(0,idxs[0][i]-2):min(idxs[0].size,idxs[0][i]+3),
            #                                                                 max(0,idxs[1][i]-2):min(idxs[1].size, idxs[1][i]+3),
            #                                                                 max(0,idxs[2][i]-2):min(idxs[2].size, idxs[2][i]+3)]))


if __name__ == "__main__":
    main()
