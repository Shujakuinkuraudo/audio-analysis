import wave
temp_file_path = "recording.wav"

with wave.open(temp_file_path, "rb") as audio:
# 获取音频的基本信息
    frame_rate = audio.getframerate()
    num_frames = audio.getnframes()
    duration = num_frames / float(frame_rate)