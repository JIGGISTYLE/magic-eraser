"""Lightweight two-stage gated model, trained from scratch on Flowers.
Contextual attention is omitted. Architecture matches the standalone training notebook.
"""
import torch
from torch import nn
import torch.nn.functional as F

class GatedConv(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.conv=nn.Conv2d(cin,2*cout,3,stride,padding=dilation,dilation=dilation)
    def forward(self,x):
        features,gate=self.conv(x).chunk(2,dim=1)
        return F.elu(features)*torch.sigmoid(gate)

class Stage(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.enc=nn.Sequential(GatedConv(5,c),GatedConv(c,2*c,2),GatedConv(2*c,4*c,2))
        self.middle=nn.Sequential(*[GatedConv(4*c,4*c,dilation=d) for d in (1,2,4,8)])
        self.up1=GatedConv(4*c,2*c); self.up2=GatedConv(2*c,c)
        self.out=nn.Conv2d(c,3,3,padding=1)
    def forward(self,x):
        z=self.middle(self.enc(x))
        z=self.up1(F.interpolate(z,scale_factor=2,mode="nearest"))
        z=self.up2(F.interpolate(z,scale_factor=2,mode="nearest"))
        return torch.tanh(self.out(z))

class DeepFillLite(nn.Module):
    def __init__(self,c=24):
        super().__init__(); self.coarse=Stage(c); self.refine=Stage(c)
    def forward(self,image,mask):
        masked=image*(1-mask)
        coarse=self.coarse(torch.cat([masked,mask,torch.ones_like(mask)],1))
        filled=masked+coarse*mask
        refined=self.refine(torch.cat([filled,mask,torch.ones_like(mask)],1))
        return coarse,refined

class DeepFillLiteGenerator(nn.Module):
    """Adapt the trained model to the shared DeepFill inference interface."""
    def __init__(self, checkpoint):
        super().__init__()
        self.network = DeepFillLite(checkpoint['channels'])
        self.network.load_state_dict(checkpoint['state_dict'], strict=True)

    def forward(self, inputs, mask):
        return self.network(inputs[:, :3], mask)
