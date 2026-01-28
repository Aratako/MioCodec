import typing as tp


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
