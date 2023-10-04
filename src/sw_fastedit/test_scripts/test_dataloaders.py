from __future__ import annotations

import monai.transforms as mt
import torch
from monai.data import ArrayDataset, DataLoader, MetaTensor

NETWORK_INPUT_SHAPE = (1, 128, 128, 256)
NUM_IMAGES = 50


def get_xy():
    xs = [256 * torch.rand(NETWORK_INPUT_SHAPE) for _ in range(NUM_IMAGES)]
    ys = [torch.rand(NETWORK_INPUT_SHAPE) for _ in range(NUM_IMAGES)]
    return xs, ys


transform = mt.Compose([mt.ToDevice(device="cpu")])


def get_data_loader():
    x, y = get_xy()
    dataset = ArrayDataset(x, seg=y, img_transform=transform, seg_transform=transform)
    loader = DataLoader(dataset, num_workers=1, batch_size=1, multiprocessing_context="spawn")
    return loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data_loader()
    for x in train_loader:
        print(type(x[0]))
