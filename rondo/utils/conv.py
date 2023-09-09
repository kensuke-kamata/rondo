import numpy as np

def get_conv_outsize(input, kernel, stride=1, pad=0):
    """
    Args:
        input  (int): The size of input feature map.
        kernel (int): The size of convolution kernel.
        stride (int): The size of stride.
        pad    (int): The size of padding.
    """
    return (input + pad * 2 - kernel) // stride + 1
