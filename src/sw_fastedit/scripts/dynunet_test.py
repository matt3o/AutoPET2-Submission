from __future__ import annotations

import torch
from monai.networks.nets import DynUNet

# Initialize DynUNet model
model = DynUNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=2,
    kernel_size=[3, 3, 3, 3, 3, 3],
    strides=[1, 2, 2, 2, 2, [2, 2, 1]],
    upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
    norm_name="instance",
    deep_supervision=False,
    res_block=True,
)

print(model)

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 128, 128, 128)


# Function to print attributes of a layer's output
def print_shape_and_attributes(layer_number, module, input, output):
    kernel_size = getattr(module, "kernel_size", "N/A")
    num_filters = getattr(module, "out_channels", "N/A")
    stride = getattr(module, "stride", "N/A")
    input_shape = input[0].shape
    print(
        f"Layer {layer_number}: {module.__class__.__name__}, Input shape: {input_shape}, Kernel size: {kernel_size}, Num filters: {num_filters}, Stride: {stride}, Output shape: {output.shape}"
    )


# Function to recursively attach hooks to all layers in a module
def register_hooks(module, parent_counter=""):
    local_counter = 1
    if not list(module.children()):
        layer_number = f"{parent_counter}{local_counter}"
        hook = module.register_forward_hook(
            lambda module, input, output, layer_number=layer_number: print_shape_and_attributes(
                layer_number, module, input, output
            )
        )
        hooks.append(hook)
    else:
        for sub_module in module.children():
            next_counter = f"{parent_counter}{local_counter}."
            register_hooks(sub_module, next_counter)
            local_counter += 1


# Attach hooks to each layer in the model
hooks = []
register_hooks(model)

# Forward propagate the dummy input to get sizes
# _ = model(dummy_input)

# Remove hooks
for hook in hooks:
    hook.remove()
