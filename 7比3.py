import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures


# 数据集路径和标签映射
DATASET_PATH = 'data/emodb'
emotion_map = {
    'W': 'anger',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral',
    'L': 'boredom'
}

# 特征提取函数
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch = np.max(pitches)
    rms = librosa.feature.rms(y=audio)
    volume = np.mean(rms)
    mfccs_high = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    timbre = np.mean(mfccs_high.T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    speech_frames = np.sum(librosa.effects.split(audio, top_db=20))
    total_frames = len(audio)
    rate_of_speech = speech_frames / total_frames
    features = np.hstack((mfccs_mean, pitch, volume, timbre, spectral_centroid, spectral_bandwidth, spectral_contrast, chroma_stft, rate_of_speech))
    return features

# 批量提取特征
def process_file(file):
    file_path = os.path.join(DATASET_PATH, file)
    emotion = emotion_map[file[5]]
    features = extract_features(file_path)
    return features, emotion

def main():
    # 并行处理文件
    files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.wav')]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, files))

    # 提取所有文件的特征和标签
    features_list, labels = zip(*results)

    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    labels_df = pd.DataFrame(labels, columns=['label'])

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels_df)

    # 特征标准化
    scaler = StandardScaler()
    # features_list = scaler.fit_transform(features_list)

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
    svm = SVC()
    random_search = RandomizedSearchCV(svm, param_grid, n_iter=50, cv=5, random_state=42)
    random_search.fit(features_list, y_encoded)
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    # 训练集和测试集划分（7比3）
    X_train, X_test, y_train, y_test = train_test_split(features_list, y_encoded, test_size=0.3, random_state=42)

    # 训练SVM分类器
    svm_classifier = SVC(**best_params, random_state=42)
    svm_classifier.fit(X_train, y_train)

    # 预测
    y_pred = svm_classifier.predict(X_test)

    # 评价模型性能
    class_names = label_encoder.inverse_transform(np.unique(y_encoded))  # 获取原始标签名
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 绘制混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # # 创建Tkinter应用
    # class SpeechEmotionRecognitionApp:
    #     def __init__(self, root):
    #         self.root = root
    #         self.root.title("Speech Emotion Recognition")
    #         self.root.geometry("864x605")

    #         self.background_image = Image.open("pic/background.jpg")
    #         self.background_photo = ImageTk.PhotoImage(self.background_image)
    #         self.background_label = tk.Label(root, image=self.background_photo)
    #         self.background_label.place(relwidth=1, relheight=1)

    #         self.label = tk.Label(root, text="Select or record an audio file to predict its emotion", bg='white')
    #         self.label.pack(pady=20)

    #         self.choose_button = tk.Button(root, text="Choose File", command=self.load_file, bg='white')
    #         self.choose_button.pack(pady=10)

    #         self.record_button = tk.Button(root, text="Record Audio", command=self.record_audio, bg='white')
    #         self.record_button.pack(pady=10)

    #         self.status_label = tk.Label(root, text="", bg='white')
    #         self.status_label.pack(pady=20)

    #     def load_file(self):
    #         file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    #         if file_path:
    #             emotion = self.predict_emotion(file_path)
    #             messagebox.showinfo("Prediction", f"The predicted emotion is: {emotion}")

    #     def record_audio(self):
    #         duration = 3  # seconds
    #         fs = 44100  # Sample rate
    #         self.status_label.config(text="Recording will start in 1 second. Please speak for 3 seconds.")
    #         self.root.update()
    #         self.root.after(1000, self.start_recording, duration, fs)

    #     def start_recording(self, duration, fs):
    #         self.status_label.config(text="开始录音")
    #         self.root.update()
    #         recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    #         sd.wait()  # Wait until recording is finished
    #         self.status_label.config(text="录音结束")
    #         self.root.update()
    #         file_path = "recorded_audio.wav"
    #         sf.write(file_path, recording, fs)
    #         emotion = self.predict_emotion(file_path)
    #         messagebox.showinfo("Prediction", f"The predicted emotion is: {emotion}")

    #     def predict_emotion(self, file_path):
    #         features = extract_features(file_path)
    #         features_scaled = scaler.transform([features])
    #         prediction = svm_classifier.predict(features_scaled)
    #         emotion = label_encoder.inverse_transform(prediction)[0]
    #         return emotion

    # root = tk.Tk()
    # app = SpeechEmotionRecognitionApp(root)
    # root.mainloop()

if __name__ == '__main__':
    main()



'''
Best parameters found:  {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
              precision    recall  f1-score   support

       anger       0.85      0.91      0.88        32
     boredom       0.94      0.63      0.76        27
     disgust       0.68      0.94      0.79        16
        fear       0.87      0.91      0.89        22
   happiness       0.80      0.59      0.68        27
     neutral       0.70      0.95      0.81        22
     sadness       1.00      0.93      0.97        15

    accuracy                           0.82       161
   macro avg       0.84      0.84      0.82       161
weighted avg       0.84      0.82      0.82       161

'''