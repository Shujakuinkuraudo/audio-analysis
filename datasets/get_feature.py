
import torch
import torchaudio
from typing import Tuple
def get_feature(wave_form:torch.Tensor, sr, mfcc_transform=None) -> Tuple[torch.Tensor]:
    wave_form -= wave_form.mean()
    wave_form /= wave_form.abs().max()
    
    frames = wave_form.unfold(0, 400, 200)
    zcr = frames.sign().diff(dim=1).ne(0).sum(dim=1).float() # 199
    energy = frames.pow(2).sum(dim=1) # 199
    max_val = frames.abs().max(dim=1).values # 199
    fft = torch.fft.rfft(frames, 20).real # 199,11
    

    mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={"n_fft": 800,"win_length":800, "hop_length": 100, "n_mels": 23}, sample_rate=sr)
    mfcc_total = mfcc_transform(wave_form).unsqueeze(0) # 13, 9
    mfcc_total = torch.concat([mfcc_total, torchaudio.transforms.AmplitudeToDB(top_db=80)(mfcc_total)], dim=0)

    frames = wave_form.unfold(0, 16000, 1000) 
    hamming = torch.hamming_window(16000)
    frames = frames * hamming
    mfcc_partial = mfcc_transform(frames).unsqueeze(1) # 59, 13, 9
    mfcc_partial = torch.concat([mfcc_partial, torchaudio.transforms.AmplitudeToDB(top_db=80)(mfcc_partial)], dim=1)
    
    return zcr, energy, mfcc_total, max_val, fft, mfcc_partial