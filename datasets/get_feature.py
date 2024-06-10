
import torch
from typing import Tuple
def get_feature(wave_form:torch.Tensor, sr, mfcc_transform=None) -> Tuple[torch.Tensor]:
    wave_form -= wave_form.mean()
    wave_form /= wave_form.abs().max()
    
    frames = wave_form.unfold(0, 400, 200)
    zcr = frames.sign().diff(dim=1).ne(0).sum(dim=1).float() # 199
    energy = frames.pow(2).sum(dim=1) # 199
    max_val = frames.abs().max(dim=1).values # 199
    fft = torch.fft.rfft(frames, 20).real # 199,11
    
    mfcc_total = mfcc_transform(wave_form).unsqueeze(0) # 13, 9
    frames = wave_form.unfold(0, 1600, 800) 
    hamming = torch.hamming_window(1600)
    frames = frames * hamming
    mfcc_partial = mfcc_transform(frames) # 59, 13, 9
    
    return zcr, energy, mfcc_total, max_val, fft, mfcc_partial