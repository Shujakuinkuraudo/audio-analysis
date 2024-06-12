
import torch
import torchaudio
from typing import Tuple
import librosa
def get_feature(wave_form:torch.Tensor, sr, mfcc_transform=None, embed_len=39) -> Tuple[torch.Tensor]:
    wave_form -= wave_form.mean()
    wave_form /= wave_form.abs().max()
    
    frames = wave_form.unfold(0, 400, 200)
    zcr = frames.sign().diff(dim=1).ne(0).sum(dim=1).float() # 199
    energy = frames.pow(2).sum(dim=1) # 199
    max_val = frames.abs().max(dim=1).values # 199
    fft = torch.fft.rfft(frames, 20).real # 199,11
    

    # mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=embed_len, sample_rate=sr)
    # mfcc_total = mfcc_transform(wave_form) # 13, 199
    mfcc_total = torch.tensor(librosa.feature.mfcc(y=wave_form.numpy(), sr=sr, n_mfcc=embed_len)).T
    # mfcc_total = torch.concat([mfcc_total, torchaudio.transforms.AmplitudeToDB(top_db=80)(mfcc_total)], dim=0)

    frames = wave_form.unfold(0, 22050, 4000) 
    hamming = torch.hamming_window(22050)
    frames = frames * hamming
    mfcc_partial = torch.tensor(librosa.feature.mfcc(y=frames.numpy(), sr=sr, n_mfcc=embed_len)).permute(0, 2, 1)
    # mfcc_partial = torch.concat([mfcc_partial, torchaudio.transforms.AmplitudeToDB(top_db=80)(mfcc_partial)], dim=1)
    
    return zcr, energy, mfcc_total, max_val, fft, mfcc_partial
if __name__ == "__main__":
    wave_form = torch.randn(16000*10)
    sr = 16000
    zcr, energy, mfcc_total, max_val, fft, mfcc_partial = get_feature(wave_form, sr)
    print(zcr.shape, mfcc_total.shape, mfcc_partial.shape)
    
