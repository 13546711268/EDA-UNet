import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, C, _, _ = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        grad_weight = (grad_output * y).sum(dim=(2, 3))
        grad_bias = grad_output.sum(dim=(2, 3))
        return gx, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class EDABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, bias=True),
        )
        self.ssa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = self.ssa(torch.cat([avg_out, max_out], dim=1))
        x = self.sg(x)
        x = x * w
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class EDANet(nn.Module):
    def __init__(
        self,
        num_classes=5,
        in_channels=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=None,
        dec_blk_nums=None,
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = []
        if dec_blk_nums is None:
            dec_blk_nums = []
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, num_classes, kernel_size=3, padding=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[EDABlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[EDABlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[EDABlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def _pad(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp):
        _, _, H, W = inp.shape
        inp = self._pad(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(encs)):
            x = up(x)
            x = x + skip
            x = decoder(x)
        x = self.ending(x)
        return x[:, :, :H, :W]