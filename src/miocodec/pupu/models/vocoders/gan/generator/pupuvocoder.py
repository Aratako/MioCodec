import math

import julius
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn import Conv1d
from torch.nn.utils import remove_weight_norm, weight_norm

from miocodec.pupu.modules.activation_functions import ADAASnakeBeta
from miocodec.pupu.modules.anti_aliasing import Activation1d
from miocodec.pupu.modules.vocoder_blocks import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.cfg = cfg

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [
                Activation1d(activation=ADAASnakeBeta(channels))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResampleUpsampler(nn.Module):
    def __init__(self, n_mel, upsample_rate, c_prev, c_cur):
        super(ResampleUpsampler, self).__init__()
        self.scale_factor = upsample_rate
        self.convolution_after = weight_norm(Conv1d(c_prev, c_cur, 1, 1))
        self.convolution_noise = weight_norm(Conv1d(n_mel, c_prev, 7, 1, padding=3))

    def forward(self, x, x0, upps):
        B, C, T = x0.shape
        y0 = torch.zeros(B, C, T * upps, device=x0.device)
        y0[:, :, ::upps] = x0
        y0 = self.convolution_noise(y0)
        with autocast(y0.device.type, enabled=False):
            y0 = julius.highpass_filter(y0.float(), 0.5 / self.scale_factor)

        B, C, T = x.shape
        y = torch.zeros(B, C, T * self.scale_factor, device=x.device)
        y[:, :, :: self.scale_factor] = x
        with autocast(y.device.type, enabled=False):
            y = julius.lowpass_filter(y.float(), 0.5 / self.scale_factor)

        y0 = y0.to(x0.dtype)
        y = y.to(x.dtype)
        y = y + y0
        y = self.convolution_after(y)
        return y


class PupuVocoder(nn.Module):
    def __init__(self, cfg):
        super(PupuVocoder, self).__init__()

        self.cfg = cfg
        self.num_kernels = len(self.cfg.model.pupuvocoder.resblock_kernel_sizes)
        self.num_upsamples = len(self.cfg.model.pupuvocoder.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(
                self.cfg.preprocess.n_mel,
                self.cfg.model.pupuvocoder.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        resblock = ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.cfg.model.pupuvocoder.upsample_rates,
                self.cfg.model.pupuvocoder.upsample_kernel_sizes,
            )
        ):
            c_prev = self.cfg.model.pupuvocoder.upsample_initial_channel // (2**i)
            c_cur = self.cfg.model.pupuvocoder.upsample_initial_channel // (
                2 ** (i + 1)
            )
            self.ups.append(
                ResampleUpsampler(
                    self.cfg.model.pupuvocoder.upsample_initial_channel,
                    u,
                    c_prev,
                    c_cur,
                )
            )

        self.resblocks = nn.ModuleList()
        ch = self.cfg.model.pupuvocoder.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(
                zip(
                    self.cfg.model.pupuvocoder.resblock_kernel_sizes,
                    self.cfg.model.pupuvocoder.resblock_dilation_sizes,
                )
            ):
                self.resblocks.append(resblock(cfg, ch, k, d))

        self.activation_post = Activation1d(activation=ADAASnakeBeta(c_cur))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=False))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upps = [
            math.prod(self.cfg.model.pupuvocoder.upsample_rates[:-4]),
            math.prod(self.cfg.model.pupuvocoder.upsample_rates[:-3]),
            math.prod(self.cfg.model.pupuvocoder.upsample_rates[:-2]),
            math.prod(self.cfg.model.pupuvocoder.upsample_rates[:-1]),
            math.prod(self.cfg.model.pupuvocoder.upsample_rates),
        ]

    def forward(self, x):
        x = self.conv_pre(x)
        x0 = x
        for i in range(self.num_upsamples):
            x = self.ups[i](x, x0, self.upps[i])
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.clamp(x, min=-1.0, max=1.0)

        return x
