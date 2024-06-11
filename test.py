import torchaudio
import torch

mfcc_transform = torchaudio.transforms.MFCC(
    n_mfcc=13,
    melkwargs={"n_fft": 400, "win_length": 400, "hop_length": 200, "n_mels": 23},
    sample_rate=16000,
)
dummy_wave_form = torch.randn(16000 * 3)
frames = dummy_wave_form.unfold(0, 1600, 800)
window = torch.hann_window(1600)
frames = frames * window
print(mfcc_transform(frames).shape)
