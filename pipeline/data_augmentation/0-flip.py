#!/usr/bin/env python3
"""Flip"""
import tensorflow as tf


def flip_image(image):
    """def the func"""
    return tf.reverse(image, axis=1)
