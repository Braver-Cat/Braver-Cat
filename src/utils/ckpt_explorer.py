import torch
from rich import print

checkpoint = torch.load("../checkpoints/swift-eon-23/valid_but_named_weights_only.ckpt")
print(checkpoint.keys())
