import torch
import os

tkwargs = {
    "dtype": torch.double,
    "device": "cpu",
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")