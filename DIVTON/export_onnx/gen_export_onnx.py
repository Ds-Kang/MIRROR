import os
import torch
import torch.nn as nn
from models.networks import ResUnetGenerator, load_checkpoint

import torch.onnx

class GenModel(nn.Module):
    def __init__(self, gen_model):
        super(GenModel, self).__init__()
        self.gen_model = gen_model

    def forward(self, x):
        gen_inputs, parse_sparse = torch.split(x, [7, 7], 1)
        gen_output = self.gen_model(gen_inputs)

        return gen_output


model_nop = ResUnetGenerator(7, 5, 5, ngf=64, norm_layer=nn.BatchNorm2d)
load_checkpoint(model_nop, "torch_checkpoints/PFAFN_gen_epoch_101_phif_pl_skip.pth")
model = GenModel(model_nop)
model.eval()

# model sample input
x = torch.randn(1, 14, 256, 192, requires_grad=True)

with torch.no_grad():
    torch_out = model(x)

onnx_path = "onnx_checkpoints"
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

torch.onnx.export(
    model,
    x,
    f"{onnx_path}/gen.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)
