import torch
import os


SMOKE_TEST = os.environ.get("SMOKE_TEST")
DEVICE = os.environ.get("BOTORCH_DEVICE", default="cpu")

tkwargs = {
    "dtype": torch.double,
    "device": DEVICE,
}
