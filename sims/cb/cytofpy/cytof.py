import torch
import math
import copy
import datetime
import numpy as np
import advi

class Cytof(advi.Model):
    def __init__(self, priors=None, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.priors = priors

    pass


