import torch 
import numpy as np

from FastGeodis import generalised_geodesic3d
from scipy.ndimage.morphology import distance_transform_edt as distance_transform_edt_scipy

import cupy as cp
from cucim.core.operations.morphology import distance_transform_edt as distance_transform_edt_cupy

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

import time

def main():
    vector_t = torch.load("discrepancy.pt").squeeze()
#    vector_t = torch.rand((128,128,128))
    logger.info(vector_t.shape)

    before = time.time()
    iterations = 20
    vector = vector_t.clone().cpu().numpy()
    for i in range(iterations):    
        distance = distance_transform_edt_scipy(vector)
    elapsed_time = time.time() - before
    logger.info("distance_transform_edt_scipy: {} runs took {:f} seconds, which means {:f} seconds per run".format(iterations, elapsed_time, elapsed_time / iterations))
    
    
    before = time.time()
    vector_cp = cp.asarray(vector_t)
    for i in range(iterations):
        distance_cp = distance_transform_edt_cupy(vector_cp)
    elapsed_time = time.time() - before
    logger.info("distance_transform_edt_cupy: {} runs took {:f} seconds, which means {:f} seconds per run.".format(iterations, elapsed_time, elapsed_time / iterations))

    assert np.allclose(distance, distance_cp, atol=0.001)


    # d_edt_gpu = generalised_geodesic3d(vector_t,
    #                                     vector_t,
    #                                     [1.0, 1.0, 1.0],
    #                                     10e10,
    #                                     0.0,
    #                                     2)
    # distance = distance.squeeze()
    # d_edt_gpu = d_edt_gpu.squeeze()
    # logger.info("d_edt_gpu:\n {}".format(d_edt_gpu))
    # if not np.allclose(distance, d_edt_gpu, atol=0.001):
    #     logger.error(np.logical_not(np.isclose(distance, d_edt_gpu)))
    #     idxs = np.where(np.isclose(distance, d_edt_gpu) == False)
    #     assert len(idxs) > 0
    #     for i in range(0, min(5, idxs[0].size)):
    #         position = (idxs[0][i], idxs[1][i], idxs[2][i])
    #         logger.info("{} \n".format(position))
    #         logger.info("Item at position: {} which has value: {} \nscipy distance: {} , GPU d_edt_gpu: {}".format(
    #                     position, vector.squeeze()[position], distance[position], d_edt_gpu[position]))
    #         logger.info("Context array: {}".format(vector.squeeze()[max(0,idxs[0][i]-2):min(idxs[0].size,idxs[0][i]+3),
    #                                                                         max(0,idxs[1][i]-2):min(idxs[1].size, idxs[1][i]+3),
    #                                                                         max(0,idxs[2][i]-2):min(idxs[2].size, idxs[2][i]+3)]))

        # raise UserWarning("Distance transform mismatch!")



if __name__ == "__main__":
    main()