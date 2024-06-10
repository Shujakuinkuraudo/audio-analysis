from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import os
import librosa
import wave
import soundfile
import ffmpeg

app = FastAPI()
origins = [
    "http://localhost:25000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传并保存音频文件
@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    file_path = os.path.join('audio', audio.filename)  
    with open(file_path, 'wb') as f:
        f.write(await audio.read())
    file_path:str
    if file_path.endswith("mp3"):
        ffmpeg.input(file_path).output(file_path.replace("mp3","wav")).overwrite_output().run_async(pipe_stdin=True)
    return process_data(file_path)
    

def process_data(audio_file):
    data, samplerate = librosa.load(audio_file.replace("mp3","wav"))
    return classify(data)

def classify(data):
    import random
    return {"emo":random.random(), "ac":random.random()}
    
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# import librosa
# import aiofiles 
# import os

# app = FastAPI()

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         print(file)
#         file_location = os.path.join(UPLOAD_DIR, file.filename)
        
#         async with aiofiles.open(file_location, 'wb') as out_file:
#             while content := await file.read(1024):
#                 await out_file.write(content)
        
#         # Load the audio file using librosa
#         y, sr = librosa.load(file_location, sr=None)
        
#         # Perform some audio processing (e.g., get duration)
#         duration = librosa.get_duration(y=y, sr=sr)
        
#         return JSONResponse(content={"filename": file.filename, "duration": duration})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import uvicorn

# app = FastAPI()

# @app.post("/upload-audio")
# async def upload_audio(file: UploadFile = File(...)):
#     try:
#         content = await file.read()
#         # 在这里处理音频文件，例如保存文件或进行分析
#         with open('uploaded_recording.wav', 'wb') as f:
#             f.write(content)
#         return JSONResponse(content={"message": "File uploaded successfully"}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"message": str(e)}, status_code=500)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)