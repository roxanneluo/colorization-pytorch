import numpy as np
import torch
from PIL import ImageFilter


class RandomGaussianBlur(object):
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, image):
        radius = np.abs(np.random.normal(scale=self.stddev))
        return image.filter(ImageFilter.GaussianBlur(radius))


class RandomGaussianNoise(object):
    """
    Generate random noise with random stddev
    """
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, image, amin=-0.5, amax=0.5):
        # image is a tensor
        sigma = np.abs(np.random.normal(scale=self.stddev))
        return (image + sigma * torch.randn_like(image)).clamp(amin, amax)

