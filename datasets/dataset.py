import torch
from typing import Tuple
import tqdm
class dataset:
    def get_feature(self, wave_form:torch.Tensor, sr) -> Tuple[torch.Tensor]:
        frames = wave_form.unfold(0, 400, 200)
        zcr = frames.sign().diff(dim=1).ne(0).sum(dim=1).float() # 199
        energy = frames.pow(2).sum(dim=1) # 199
        max_val = frames.abs().max(dim=1).values # 199
        fft = torch.fft.rfft(frames, 20).real # 199,11
        
        mfcc_total = torch.mean((wave_form).unsqueeze(0)) # 1, 13, 201
        frames = wave_form.unfold(0, 1600, 800) 
        hamming = torch.hamming_window(1600)
        frames = frames * hamming
        mfcc_partial = self.mfcc_transform(frames) # 59, 13, 9
        
        return zcr, energy, mfcc_total, max_val, fft, mfcc_partial
    def get_feature_data(self):
        datas = []
        targets = []

        for zcr, energy, mfcc_total, max_val, fft,mfcc_partial, target in tqdm.tqdm(self):
            datas.append(torch.cat([zcr, energy, max_val, fft.view(-1)], dim=0))
            targets.append(target)
        return datas, targets