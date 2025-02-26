import torch
import torch.nn as nn
from models.model_4channel import AWNet


class wrapped_4_channel(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = AWNet(4, 3, block=[3, 3, 3, 4, 4])

    def forward(self, x):
        return self.module(x)


# Dimension is divisible by utils.dwt_init()
sample_input = torch.rand((1, 4, 896, 1984), device="cuda")

model = wrapped_4_channel()

checkpoint = torch.load("best_weight/best_4channel.pkl", map_location="cuda")
model.load_state_dict(checkpoint["model_state"], strict=True)
model.cuda().eval()
print("OK")
torch.onnx.export(
    model,  # PyTorch Model
    (sample_input),  # Input tensor
    "tmp_awnet.onnx",  # Output file (eg. 'output_model.onnx')
    opset_version=12,  # Operator support version
    input_names=["input"],  # Input tensor name (arbitary)
    output_names=["output"],  # Output tensor name (arbitary)
    do_constant_folding=False,
)
