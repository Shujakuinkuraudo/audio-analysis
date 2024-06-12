def extract_features(file_path):
    import librosa
    import numpy as np
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')

    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch = np.max(pitches) # 1

    rms = librosa.feature.rms(y=audio)
    volume = np.mean(rms) # 1
    
    mfccs_high = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    timbre = np.mean(mfccs_high.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0) # 7
    
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0) # 12

    speech_frames = np.sum(librosa.effects.split(audio, top_db=20))
    total_frames = len(audio)
    rate_of_speech = speech_frames / total_frames # 1


    features = np.hstack(( pitch, volume, timbre, spectral_centroid, spectral_bandwidth, spectral_contrast, chroma_stft, rate_of_speech))
    return features

if __name__ == "__main__":
    file_path = "data/emodb/03a01Nc.wav"
    print(extract_features(file_path).shape)