#!/usr/bin/env python3

"""performs a random crop of an image"""


def crop_image(image, size):
    """
    Performs a random crop on an image.
     """
    return tf.image.random_crop(image, size)
