from __future__ import annotations

import torch
from monai.losses import DiceLoss
from monai.metrics import compute_dice

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

actual_patch_y = torch.tensor([[0, 0], [1, 0]]).unsqueeze(0).unsqueeze(0)
actual_patch_y_pred = torch.tensor([[1, 1], [1, 1]]).unsqueeze(0).unsqueeze(0)


# actual_patch_y = torch.zeros((2, 128,128,128)).to(device="cuda:0")
# actual_patch_y_pred = torch.zeros((2, 128,128,128)).to(device="cuda:0")
# print(actual_patch_y_pred.shape)
#
# actual_patch_y_pred[1,0,0,0] = 1
dice_loss = DiceLoss(include_background=False)

loss = dice_loss.forward(input=actual_patch_y_pred, target=actual_patch_y)
print(f"DiceLoss: {loss}")

score = compute_dice(y_pred=actual_patch_y_pred, y=actual_patch_y, include_background=True)
print(f"compute_dice score: {score}")

# actual_patch_y[1,0,0,0] = 1
# print("Setting a label entry to 1")
# loss = dice_loss.forward(input=actual_patch_y_pred.unsqueeze(0), target=actual_patch_y.unsqueeze(0)).item()
# print(f"DiceLoss: {loss}")

# score = compute_dice(y_pred=actual_patch_y_pred.unsqueeze(0), y=actual_patch_y.unsqueeze(0), include_background=True)
# print(f"compute_dice score: {score}")
