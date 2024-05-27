import torch
import tqdm
from torch.utils.data import Dataset
import torchaudio
from typing import List,Literal, Tuple
from .dataset import dataset
from torchvision import transforms
import glob
class savee_dataset(Dataset, dataset):
    emotions = ["a", "d", "f", "h", "n", "sa", "su"]
    def __init__(self, root: str = "data/savee", train=True, leave_out_people_id: List[int] = [], sr = 16000, win_length = 200, hop_length = 100, n_fft=400):
        self.sr = sr
        self.emo_dict = {self.emotions[i]:i for i in range(len(self.emotions))}
        self.train = train
        self.people_id = ["DC", "JE", "JK", "KL"]
        self.time = 4

        self.data_path = self.preprocess(glob.glob(root+"/*.wav"), [self.people_id[i] for i in leave_out_people_id])
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=23, melkwargs={"n_fft": n_fft,"win_length":win_length, "hop_length": hop_length, "n_mels": 40}, sample_rate=sr)
        self.data = self.get_data()
    
    def get_data(self):
        data = []
        for index in range(len(self)):
            wave_form, sr = torchaudio.load(self.data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * self.time:
                wave_form = torch.cat((wave_form, torch.zeros(1, self.sr*self.time-wave_form.shape[1])), dim=1)
            if wave_form.shape[1] > self.sr*self.time:
                wave_form = wave_form[:,:self.sr*self.time]

            wave_form = wave_form.mean(dim=0)
            wave_form /= wave_form.abs().topk(int(0.02 * self.sr * self.time)).values.mean()
            
            target = self.emo_dict[self.data_path[index].split("/")[-1].split("_")[1][:-6]]
            data.append([*self.get_feature(wave_form, sr),wave_form.view(1,-1), target])
        return data
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        return self.data[index]
    
    def __len__(self):
        return len(self.data_path)

        
    def preprocess(self, data_paths, leave_out_people):
        data = []
        for data_path in data_paths:
            if self.train:
                if data_path.split("/")[-1][:2] not in leave_out_people:
                    data.append(data_path)
            else:
                if data_path.split("/")[-1][:2] in leave_out_people:
                    data.append(data_path)
        
        return data
    

        
def savee_fold_dl(root: str= "data/savee", fold: int = 5, sr = 16000, n_fft=400, hop_length=100, win_length = 200):
    each_peole = 4 // fold
    leave_out_peole = [[j + i for i in range(each_peole)] for j in range(0, 4 - each_peole + 1, each_peole)]
    return [[savee_dataset(root, train=True, leave_out_people_id=leave_out_peole[i], sr=sr, win_length=win_length, hop_length=hop_length, n_fft=n_fft),savee_dataset(root, train=False, leave_out_people_id=leave_out_peole[i], sr=sr, win_length=win_length, hop_length=hop_length, n_fft=n_fft)] for i in range(fold)], savee_dataset.emotions

def savee_fold_ml(root: str= "data/savee", fold: int = 5, sr = 16000):
    each_peole = 4 // fold
    leave_out_peole = [[j + i for i in range(each_peole)] for j in range(0, 4 - each_peole + 1, each_peole)]
    return [[savee_dataset(root, train=True, leave_out_people_id=leave_out_peole[i], sr=sr).get_feature_data(),savee_dataset(root, train=False, leave_out_people_id=leave_out_peole[i], sr=sr).get_feature_data()] for i in range(fold)], savee_dataset.emotions


        
if __name__ == "__main__":
    # train_dataset = savee_dataset("data/savee", train=True, leave_out_people=["03"])
    # test_dataset = savee_dataset("data/savee", train=False, leave_out_people=["03"])
    

    # train_datas = []
    # train_targets = []
    # import tqdm
    # for zcr, energy, mfcc, max_val, fft, target in tqdm.tqdm(train_dataset):
    #     train_datas.append(torch.concat([zcr, energy, max_val, fft.view(-1)], dim=0))
    #     train_targets.append(target)
    # test_datas = []
    # test_targets = []
    # import tqdm
    # for zcr, energy, mfcc, max_val, fft, target in tqdm.tqdm(test_dataset):
    #     test_datas.append(torch.concat([zcr, energy, max_val, fft.view(-1)], dim=0))
    #     test_targets.append(target)
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.metrics import accuracy_score
    # clf = DecisionTreeClassifier()
    # clf.fit(train_datas, train_targets)
    # print(accuracy_score(clf.predict(test_datas), test_targets))

    print(savee_fold_ml("data/savee",4)[0])