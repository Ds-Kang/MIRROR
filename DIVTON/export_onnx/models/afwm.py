import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import time

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = grid_list[::-1]

    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]

    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, sizes[::-1])] 

    return torch.stack(grid_list, dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                         ResBlock(out_chns),
                                         ResBlock(out_chns))
            
            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)

        
        return encoder_features

class RefinePyramid(nn.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)

        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

        
        mod_list = []
        for mod1, mod2 in zip(self.adaptive, self.smooth):
            mod_list.append(nn.Sequential(mod1, mod2))
        self.seq_module = nn.ModuleList(mod_list)


    def forward(self, x: List[torch.Tensor]):
        conv_ftr_list = x

        feature_list = []
        last_feature = None

        for i, mod in enumerate(self.seq_module):
            feature = mod[0](conv_ftr_list[::-1][i])

            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2.0, mode='nearest')

            feature = mod[1](feature)
            last_feature = feature
            feature_list.append(feature)

        return feature_list[::-1]

class CorrelationLayer(nn.Module):
    def __init__(self):
        super(CorrelationLayer, self).__init__()

    def forward(self, x, y):
        input_size = x.shape
        H = input_size[2]
        W = input_size[3]

        out_list = []

        for j in range(7):
            for i in range(7):
                shiftX = i - 3
                shiftY = j - 3

                if shiftX >= 0 and shiftY >= 0:
                    temp_out = F.pad(x, (shiftX, 0, shiftY, 0)) * F.pad(
                        y, (0, shiftX, 0, shiftY)
                    )
                    temp_out = torch.sum(temp_out, (0, 1))
                    temp_out = temp_out[shiftY:, shiftX:]

                    out_list.append(temp_out)

                if shiftX >= 0 and shiftY < 0:
                    temp_out = F.pad(x, (shiftX, 0, 0, -shiftY)) * F.pad(
                        y, (0, shiftX, -shiftY, 0)
                    )
                    temp_out = torch.sum(temp_out, (0, 1))
                    temp_out = temp_out[:shiftY, shiftX:]

                    out_list.append(temp_out)

                if shiftX < 0 and shiftY >= 0:
                    temp_out = F.pad(x, (0, -shiftX, shiftY, 0)) * F.pad(
                        y, (-shiftX, 0, 0, shiftY)
                    )
                    temp_out = torch.sum(temp_out, (0, 1))
                    temp_out = temp_out[shiftY:, :shiftX]

                    out_list.append(temp_out)

                if shiftX < 0 and shiftY < 0:
                    temp_out = F.pad(x, (0, -shiftX, 0, -shiftY)) * F.pad(
                        y, (-shiftX, 0, -shiftY, 0)
                    )
                    temp_out = torch.sum(temp_out, (0, 1))
                    temp_out = temp_out[:shiftY, :shiftX]

                    out_list.append(temp_out)

        out = torch.stack(out_list)
        return out.unsqueeze(0)


class LeakyReluLayer(nn.Module):
    def __init__(self, inplace=False, negative_slope=0.1):
        super(LeakyReluLayer, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        neg_x = self.negative_slope * x

        return torch.max(neg_x, x)


class AFlowSkipNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowSkipNet, self).__init__()
        self.netMain = []
        self.netRefine = []
        for i in range(num_pyramid):
            netMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                LeakyReluLayer(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )
            self.netMain.append(netMain_layer)
            self.netRefine.append(netRefine_layer)

        self.netMain = nn.ModuleList(self.netMain)
        self.netRefine = nn.ModuleList(self.netRefine)

        
        mod_list = []
        for mod1, mod2 in zip(self.netMain, self.netRefine):
            mod_list.append(nn.Sequential(mod1, mod2))
        self.seq_module = nn.ModuleList(mod_list)

        self.correlation = CorrelationLayer()

    def forward(
        self, x: torch.Tensor, x_warps: List[torch.Tensor], x_conds: List[torch.Tensor]
    ):
        last_flow = None

        for i, mod in enumerate(self.seq_module):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            if i < 3:
                if last_flow is not None:
                    x_warp_after = F.grid_sample(
                        x_warp,
                        last_flow.detach().permute(0, 2, 3, 1),
                        mode="bilinear",
                        padding_mode="border",
                    )
                else:
                    x_warp_after = x_warp

                in_shape = x_warp_after.shape

                tenCorrelation = LeakyReluLayer(inplace=False, negative_slope=0.1)(
                    self.correlation(x_warp_after, x_cond) / in_shape[1]
                )

                flow = mod[0](tenCorrelation)
                flow = apply_offset(flow)

                if last_flow is not None:
                    flow = F.grid_sample(
                        last_flow, flow, mode="bilinear", padding_mode="border"
                    )
                else:
                    flow = flow.permute(0, 3, 1, 2)

                last_flow = flow
            x_warp = F.grid_sample(
                x_warp,
                last_flow.permute(0, 2, 3, 1),
                mode="bilinear",
                padding_mode="border",
            )
            concat = torch.cat([x_warp, x_cond], 1)
            flow = mod[1](concat)
            flow = apply_offset(flow)
            flow = F.grid_sample(
                last_flow, flow, mode="bilinear", padding_mode="border"
            )

            last_flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear")

        return last_flow


class AFWM(nn.Module):
    def __init__(self, opt, input_nc):
        super(AFWM, self).__init__()
        num_filters = [64, 128, 256, 256, 256]

        self.image_features = FeatureEncoder(3, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)
        self.aflow_net = AFlowSkipNet(len(num_filters))

    def forward(self, cond_input, image_input):
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))
        image_pyramids = self.image_FPN(self.image_features(image_input))
        last_flow = self.aflow_net(image_input, image_pyramids, cond_pyramids)
        return last_flow
