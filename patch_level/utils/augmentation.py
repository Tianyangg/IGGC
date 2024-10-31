# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21-11-10 下午3:44
# @Author  : tiannyang
# @File    : augmentation.py
# Comments : 2d image augmentation sequence

import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.2)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),

    # Strengthen or weaken the contrast in each image.
    iaa.Sometimes(
        0.6,
        iaa.LinearContrast((0.75, 1.5))
    ),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.Sometimes(
        1,
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 1*255), per_channel=0.5),
    ),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Sometimes(
        0.8,
        iaa.Multiply((0.6, 1.4), per_channel=0.5),
    ),

    
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(
        0.8,
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-15, 5),
            shear=(-8, 8)
        )
    )

], random_order=True) # apply augmenters in random order


def augmentation(pil_im):

    im_arr = np.array(pil_im)  # (H, W, C=3)
    im_aug = seq(image=im_arr)

    return Image.fromarray(im_aug)
