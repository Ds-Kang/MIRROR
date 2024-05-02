import os
import torch
import torch.onnx

from models.parsing.AugmentCE2P import resnet101
import torch.nn as nn
import torch.nn.functional as F


class ParseModel(nn.Module):
    def __init__(self, parse_model):
        super(ParseModel, self).__init__()
        self.parse_model = parse_model

    def forward(self, frame):
        parse_out = self.parse_model(frame)

        logits_result = F.interpolate(
            parse_out, size=(512, 512), mode="bilinear", align_corners=True
        )
        parsing_frame = torch.argmax(logits_result, axis=1)
        crop_frame = parsing_frame[..., 112:400]

        return crop_frame.squeeze()


state_dict = torch.load("torch_checkpoints/exp-schp-201908270938-pascal-person-part.pth")[
    "state_dict"
]

parse_model = resnet101(num_classes=7, pretrained=None)
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
parse_model.load_state_dict(new_state_dict)
parse_model.eval()

parse_model_aug = ParseModel(parse_model)

# model sample input
x = torch.randn(1, 3, 512, 512, requires_grad=True)
with torch.no_grad():
    torch_out = parse_model_aug(x)

onnx_path = "onnx_checkpoints"
if not os.path.exists(onnx_path):
    os.makedirs(onnx_path)

torch.onnx.export(
    parse_model_aug,
    x,
    f"{onnx_path}/parsing.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)