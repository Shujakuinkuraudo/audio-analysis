import torch
from typing import Tuple
import torchaudio
import numpy as np
import librosa
import tqdm
class dataset:
    def zcr(self,data,frame_length,hop_length):
        zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    def rmse(self,data,frame_length=1024,hop_length=256):
        rmse=librosa.feature.rms(y = data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)
    def mfcc(self,data,sr,frame_length=1024,hop_length=256,flatten:bool=True):
        mfcc=librosa.feature.mfcc(y = data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)
    def extract_features(self,data,sr=22050,frame_length=1024,hop_length=256):
        result=np.array([])
        
        result=np.hstack((result,
                        self.zcr(data,frame_length,hop_length),
                        self.rmse(data,frame_length,hop_length),
                        self.mfcc(data,sr,frame_length,hop_length)
                        ))
        return result
    
    # NOISE
    def noise(self,data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    # STRETCH
    def stretch(self,data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)
    # SHIFT
    def shift(self,data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)
    # PITCH
    def pitch(self,data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps=pitch_factor) 
    
    
    def get_feature(self, data:torch.Tensor, sr) -> torch.Tensor:
        aud=self.extract_features(data)
        
        noised_audio=self.noise(data)
        aud2=self.extract_features(noised_audio)
        
        pitched_audio=self.pitch(data,sr)
        aud3=self.extract_features(pitched_audio)
        
        pitched_audio1=self.pitch(data,sr)
        pitched_noised_audio=self.noise(pitched_audio1)
        aud4=self.extract_features(pitched_noised_audio)
        
        return torch.tensor(aud).view(1,-1), torch.tensor(aud2).view(1,-1), torch.tensor(aud3).view(1,-1), torch.tensor(aud4).view(1,-1)

    def get_feature_data(self):
        datas = []
        targets = []

        for zcr, energy, mfcc_total, max_val, fft,mfcc_partial,wave_form, target in tqdm.tqdm(self):
            datas.append(torch.cat([zcr, energy, max_val, fft.view(-1)], dim=0))
            targets.append(target)
        return datas, targets
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        return self.data[index]
    

    def __len__(self):
        return len(self.data_path) * 4