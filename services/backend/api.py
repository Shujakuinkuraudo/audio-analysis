from fastapi import FastAPI, File, UploadFile
import torchaudio
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import Tuple


from model.CNN import CNN
model_emodb = CNN(num_classes=7)
model_emodb.load_state_dict(torch.load('model/TIM_NET_emodb.pt'))
model_emodb.eval()

model_savee = CNN(num_classes=7)
model_savee.load_state_dict(torch.load('model/TIM_NET_savee.pt'))
model_savee.eval()

model_all = CNN(num_classes=8)
model_all.load_state_dict(torch.load('model/TIM_NET_all.pt'))
model_all.eval()



import os
import librosa
import wave
import soundfile
import ffmpeg

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    frames = wave_form.unfold(0, 16000, 1000) 
    hamming = torch.hamming_window(16000)
    frames = frames * hamming
    # mfcc_partial = mfcc_transform(frames).unsqueeze(1) # 59, 13, 9
    mfcc_partial = torch.tensor(0)
    # mfcc_partial = torch.concat([mfcc_partial, torchaudio.transforms.AmplitudeToDB(top_db=80)(mfcc_partial)], dim=1)
    
    return zcr, energy, mfcc_total, max_val, fft, mfcc_partial
# 上传并保存音频文件
@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    file_path = os.path.join('audio', audio.filename)  
    with open(file_path, 'wb') as f:
        f.write(await audio.read())
    file_path:str
    if file_path.endswith("mp3"):
        ffmpeg.input(file_path).output(file_path.replace("mp3","wav")).overwrite_output().run_async(pipe_stdin=True)
    return emodb(file_path.replace("mp3","wav"))


    

@app.post("/upload_emodb")
async def upload_audio_emodb(audio: UploadFile = File(...)):
    file_path = os.path.join('audio', audio.filename)  
    with open(file_path, 'wb') as f:
        f.write(await audio.read())
    file_path:str
    if file_path.endswith("mp3"):
        ffmpeg.input(file_path).output(file_path.replace("mp3","wav")).overwrite_output().run_async(pipe_stdin=True)
    return emodb(file_path.replace("mp3","wav"))
def emodb(file_path):
    emodb_dict = {
        0 :"Anger", # Anger
        1: "Bordem",
        2: "Disgust",
        3: "Anxiety",
        4: "Happiness",
        5: "Sadness",
        6: "Neutral"
    }

    wave_form, sr = torchaudio.load(file_path, format="wav")
    standsr = 22050
    standtime = 5
    if sr != standsr:
        wave_form = torchaudio.transforms.Resample(sr, standsr)(wave_form)
        sr = standsr
        
    if wave_form.shape[1] < standsr * standtime:
        wave_form = torch.cat((wave_form, torch.zeros(1, standsr * standtime-wave_form.shape[1])), dim=1)
    if wave_form.shape[1] > standsr * standtime:
        wave_form = wave_form[:,:standsr * standtime]

    wave_form = wave_form.mean(dim=0)
    zcr, energy, mfcc_total, max_val, fft, mfcc_partial = get_feature(wave_form, sr)
    ans = model_emodb(mfcc_total.unsqueeze(0), mfcc_partial.unsqueeze(0))[0]
    ans = torch.nn.functional.softmax(ans, dim=0)
    d = {}
    for i in range(7):
        d[f"id_{i}"] = ans[i].item()
    return d


@app.post("/upload_savee")
async def upload_audio_savee(audio: UploadFile = File(...)):
    file_path = os.path.join('audio', audio.filename)  
    with open(file_path, 'wb') as f:
        f.write(await audio.read())
    file_path:str
    if file_path.endswith("mp3"):
        ffmpeg.input(file_path).output(file_path.replace("mp3","wav")).overwrite_output().run_async(pipe_stdin=True)
    return savee(file_path.replace("mp3","wav"))
def savee(file_path):
    savee_dict = {
        0 :"Anger", # Anger
        1: "Surprise",
        2: "Disgust",
        3: "Anxiety",
        4: "Happiness",
        5: "Sadness",
        6: "Neutral"
    }

    wave_form, sr = torchaudio.load(file_path, format="wav")
    standsr = 22050
    standtime = 5
    if sr != standsr:
        wave_form = torchaudio.transforms.Resample(sr, standsr)(wave_form)
        sr = standsr
        
    if wave_form.shape[1] < standsr * standtime:
        wave_form = torch.cat((wave_form, torch.zeros(1, standsr * standtime-wave_form.shape[1])), dim=1)
    if wave_form.shape[1] > standsr * standtime:
        wave_form = wave_form[:,:standsr * standtime]

    wave_form = wave_form.mean(dim=0)
    zcr, energy, mfcc_total, max_val, fft, mfcc_partial = get_feature(wave_form, sr)
    ans = model_savee(mfcc_total.unsqueeze(0), mfcc_partial.unsqueeze(0))[0]
    ans = torch.nn.functional.softmax(ans, dim=0)
    d = {}
    for i in range(7):
        d[f"id_{i}"] = ans[i].item()
    return d

@app.post("/upload_all")
async def upload_audio_all(audio: UploadFile = File(...)):
    file_path = os.path.join('audio', audio.filename)  
    with open(file_path, 'wb') as f:
        f.write(await audio.read())
    file_path:str
    if file_path.endswith("mp3"):
        ffmpeg.input(file_path).output(file_path.replace("mp3","wav")).overwrite_output().run_async(pipe_stdin=True)
    return all(file_path.replace("mp3","wav"))
def all(file_path):
    all_dict = {
        0 :"Anger", # Anger
        1: "Bordem",
        2: "Disgust",
        3: "Anxiety",
        4: "Happiness",
        5: "Sadness",
        6: "Neutral",
        7: "Surprise"
    }

    wave_form, sr = torchaudio.load(file_path, format="wav")
    standsr = 22050
    standtime = 5
    if sr != standsr:
        wave_form = torchaudio.transforms.Resample(sr, standsr)(wave_form)
        sr = standsr
        
    if wave_form.shape[1] < standsr * standtime:
        wave_form = torch.cat((wave_form, torch.zeros(1, standsr * standtime-wave_form.shape[1])), dim=1)
    if wave_form.shape[1] > standsr * standtime:
        wave_form = wave_form[:,:standsr * standtime]

    wave_form = wave_form.mean(dim=0)
    zcr, energy, mfcc_total, max_val, fft, mfcc_partial = get_feature(wave_form, sr)
    ans = model_all(mfcc_total.unsqueeze(0), mfcc_partial.unsqueeze(0))[0]
    ans = torch.nn.functional.softmax(ans, dim=0)
    d = {}
    for i in range(8):
        d[f"id_{i}"] = ans[i].item()
    return d