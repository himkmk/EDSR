"""
for FFT(Fast Fourier Transform) Losses
--> phase = FFT.angle
--> magnitude = FFT.abs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FFT(nn.Module):
    
    def __init__(self, _, loss_type):
        super(FFT, self).__init__()
        self.loss_type = loss_type


    def forward(self, sr, hr):

        fft_hr = torch.fft.fft2(hr.detach(), dim=(-2,-1))
        fft_sr = torch.fft.fft2(sr, dim=(-2,-1))

        loss = 0
        if self.loss_type.find("PHASE"):
            loss += 0.5 * F.mse_loss(torch.angle(fft_sr), torch.angle(fft_hr))
        if self.loss_type.find("MAGNITUDE"):
            loss += 0.5 * F.mse_loss(torch.abs(fft_sr), torch.abs(fft_hr))
        return loss


