#!/usr/bin/env python3
"""Flip"""
import tensorflow as tf


def flip_image(image):
    """def the func"""
    return tf.image.flip_left_right(image)
