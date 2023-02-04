def padding(in_channels: int, kernel_size: int, stride=1,):
    new_size = in_channels // stride + 1
    padding_both = (new_size - 1) * stride + kernel_size - in_channels
    return padding_both // 2