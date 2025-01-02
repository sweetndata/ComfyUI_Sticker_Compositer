import os
import torch
from PIL import Image
import torchvision.transforms.functional as tf
import logging

import numpy as np
import scipy.ndimage
import comfy.utils

MAX_RESOLUTION=16384

def composite(destination, source, x, y, mask = None, multiplier = 8, resize_source = False):
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")

    source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

    x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
    y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))

    left, top = (x // multiplier, y // multiplier)
    right, bottom = (left + source.shape[3], top + source.shape[2],)

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
        mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

    # calculate the bounds of the source that will be overlapping the destination
    # this prevents the source trying to overwrite latent pixels that are out of bounds
    # of the destination
    visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

    mask = mask[:, :, :visible_height, :visible_width]
    inverse_mask = torch.ones_like(mask) - mask

    source_portion = mask * source[:, :, :visible_height, :visible_width]
    destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]

    print('source_portion:', source_portion.shape)
    print('destination_portion:', destination_portion.shape)
    destination[:, :, top:bottom, left:right] = source_portion + destination_portion
    return destination

class StickerMaskComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x1": ("INT", {"default": 94, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y1": ("INT", {"default": 700, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x2": ("INT", {"default": 614, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y2": ("INT", {"default": 700, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x3": ("INT", {"default": 1134, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y3": ("INT", {"default": 700, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x4": ("INT", {"default": 94, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y4": ("INT", {"default": 1420, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x5": ("INT", {"default": 614, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y5": ("INT", {"default": 1420, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "x6": ("INT", {"default": 1134, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y6": ("INT", {"default": 1420, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image"

    def composite(self, destination, source, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, resize_source, mask = None):
        # print(source.shape)
        size_pack = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
        # print("dest", destination.shape)
        for i in range(len(size_pack)):
            destination = destination.clone().movedim(-1, 1)
            # print("source", source[i].unsqueeze(0).movedim(-1, 1).shape)
            destination = composite(destination, source[i].unsqueeze(0).movedim(-1, 1), size_pack[i][0], size_pack[i][1], mask[i].unsqueeze(0), 1, resize_source).movedim(1, -1)
        return (destination,)



NODE_CLASS_MAPPINGS = {
    "Sticker_Compositer": StickerMaskComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sticker_Compositer": "Sticker Compositer",
}
