import numpy as np

def get_conv_outsize(size, kernel, stride, pad):
    """
    Args:
        size   (int): The size of input feature map.
        kernel (int): The size of convolution kernel.
        stride (int): The size of stride.
        pad    (int): The size of padding.
    """
    return (size + pad * 2 - kernel) // stride + 1

def get_deconv_outsize(size, kernel, stride, pad):
    """
    Args:
        size   (int): The size of input feature map.
        kernel (int): The size of convolution kernel.
        stride (int): The size of stride.
        pad    (int): The size of padding.
    """
    return stride * (size - 1) + kernel - 2 * pad

def pair(x):
    if isinstance(x, int):
        return x, x
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return x
    else:
        raise ValueError(x)

def im2col(img, kernel, stride, pad, to_matrix=True):
    """
    Args:
        img (ndarray)   : The input image.
        kernel (int)    : The size of convolution kernel.
        stride (int)    : The size of stride.
        pad (int)       : The size of padding.
        to_matrix (bool): If True the output shape is (N * OH * OW, C * KH * KW).
                          Otherwise, the output shape is (N, C, KH, KW, OH, OW).
    Returns:
        col (ndarray): The output matrix.
    """
    N, C, H, W = img.shape
    KH, KW = pair(kernel)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # By adding an additional (stride - 1) amount of padding to the right and bottom of the image,
    # it ensures that the kernel can start at the initial padding,
    # slide over the entire image using the specified stride, and never slide into the outermost padding.
    img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), mode='constant', constant_values=(0,))
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for h in range(KH):
        h_max = h + SH * OH
        for w in range(KW):
            w_max = w + SW * OW
            col[:, :, h, w, :, :] = img[:, :, h:h_max:SH, w:w_max:SW]

    if to_matrix:
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape((N * OH * OW, -1))

    return col

def col2im(col, shape, kernel, stride, pad, to_matrix=True):
    """
    Args:
        col (ndarray)   : The input matrix.
        shape (tuple)   : The shape of input image.
        kernel (int)    : The size of convolution kernel.
        stride (int)    : The size of stride.
        pad (int)       : The size of padding.
        to_matrix (bool): If True the input shape is (N * OH * OW, C * KH * KW).
                          Otherwise, the input shape is (N, C, KH, KW, OH, OW).
    """
    N, C, H, W = shape
    KH, KW = pair(kernel)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)

    for h in range(KH):
        h_max = h + SH * OH
        for w in range(KW):
            w_max = w + SW * OW
            img[:, :, h:h_max:SH, w:w_max:SW] += col[:, :, h, w, :, :]

    return img[:, :, PH:H + PH, PW:W + PW]
