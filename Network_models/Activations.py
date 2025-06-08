from torch import nn
import torch
import torch.nn.functional as F


class custom_1(nn.Module): 
    def __init__(self): 
        super(custom_1, self).__init__()
    
    def forward(self, x):
        return 0.5 * (x + torch.sqrt(4 + x ** 2))

class BiologicalTanh(nn.Module):
    def __init__(self, scale=1.0):
        super(BiologicalTanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        # Shift and scale tanh to be in range [0, scale]
        return self.scale * 0.5 * (torch.tanh(x) + 1)

class BoundedSoftplus(nn.Module):
    def __init__(self, upper_bound=1.0):
        super(BoundedSoftplus, self).__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        return torch.clamp(F.softplus(x), max=self.upper_bound)

class BiologicalReLU(nn.Module):
    def __init__(self, upper_bound=1.0):
        super(BiologicalReLU, self).__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        # ReLU with a saturating upper bound
        return F.relu(x+1e-6) * self.upper_bound / ((x + self.upper_bound) + 1e-6)

class Zhang(nn.Module):
    def __init__(self):
        super(Zhang, self).__init__()

    def forward(self, x):
        a = 1
        b = 1
        beta = 0.8
        c = 0.2
        returned = a * torch.log(1 + torch.exp(b * (x+c)))#**beta
        return returned

class NormedReLU(nn.Module):
    def __init__(self): 
        super(NormedReLU, self).__init__()
    
    def forward(self, x):
        r = F.relu(x + 1e-6)
        return r / (r.norm(dim=-1, keepdim=True) + 1e-6)