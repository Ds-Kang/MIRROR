import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import load_checkpoint
from models.afwm import AFWM

import torch.onnx
from torch.onnx import register_custom_op_symbolic

import torch.onnx.symbolic_helper as sym_help


# def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
#     mode = sym_help._maybe_get_const(mode, "i")
#     padding_mode = sym_help._maybe_get_const(padding_mode, "i")
#     mode_str = ["bilinear", "nearest", "bicubic"][mode]
#     padding_mode_str = ["zeros", "border", "reflection"][padding_mode]
#     align_corners = int(sym_help._maybe_get_const(align_corners, "b"))
#     return g.op(
#         "com.microsoft::GridSample",
#         input,
#         grid,
#         mode_s=mode_str,
#         padding_mode_s=padding_mode_str,
#         align_corners_i=align_corners,
#     )


# # Register custom symbolic function
# register_custom_op_symbolic("::grid_sampler", grid_sampler, 1)


class WarpModel_pi(nn.Module):
    def __init__(self, warp_model):
        super(WarpModel_pi, self).__init__()
        self.warp_model = warp_model

    def forward(self, frame, clothes, edge, parse_sparse):
        last_flow = self.warp_model(torch.cat([frame, parse_sparse], 1), clothes)

        warped_cloth = F.grid_sample(
            clothes,
            last_flow.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
        )

        warped_edge = F.grid_sample(
            edge, last_flow.permute(0, 2, 3, 1), padding_mode="zeros"
        )

        gen_inputs = torch.cat([frame, warped_cloth, warped_edge, parse_sparse], 1)

        return gen_inputs

# model sample input
x = torch.randn(1, 3, 256, 192, requires_grad=True)
y = torch.randn(1, 3, 256, 192, requires_grad=True)
z = torch.randn(1, 1, 256, 192, requires_grad=True)
w = torch.randn(1, 7, 256, 192, requires_grad=True)

model = AFWM(None, 10)
model.eval()
load_checkpoint(model, "torch_checkpoints/PFAFN_warp_epoch_101_phif_pl_skip.pth")

warp_model_aug = WarpModel_pi(model)

with torch.no_grad():
    torch_out = warp_model_aug(x, y, z, w)

onnx_path = "onnx_checkpoints"
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

torch.onnx.export(
    warp_model_aug,
    (x, y, z, w),
    f"{onnx_path}/warp.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=False,
    input_names=["input1", "input2", "input3", "input4"],
    output_names=["output"],
)
