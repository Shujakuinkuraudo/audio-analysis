import torch
import tqdm
from torch.utils.data import Dataset
import torchaudio
from typing import List,Literal, Tuple
from .dataset import dataset
import glob
class emodb_dataset(Dataset, dataset):
    emotions = ["W", "L", "E", "A", "F", "T", "N"]
    def __init__(self, root: str = "data/emodb", download: bool = True, train=True, leave_out_people_id: List[int] = [], sr = 16000):
        self.sr = sr
        self.emo_dict = {self.emotions[i]:i for i in range(len(self.emotions))}
        self.train = train
        self.people_id = ["03","08","09","10","11","12","13","14","15","16"]

        self.data_path = self.preprocess(glob.glob(root+"/*.wav"), [self.people_id[i] for i in leave_out_people_id])
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={"n_fft": 400,"win_length":200, "hop_length": 100, "n_mels": 23}, sample_rate=sr)
        self.data = self.get_data()
    
    def get_data(self):
        data = []
        for index in range(len(self)):
            time = 4
            wave_form, sr = torchaudio.load(self.data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * time:
                wave_form = torch.cat((wave_form, torch.zeros(1, self.sr * time-wave_form.shape[1])), dim=1)
            if wave_form.shape[1] > self.sr * time:
                wave_form = wave_form[:,:self.sr * time]

            wave_form = wave_form.mean(dim=0)
            
                
            target = self.emo_dict[self.data_path[index].split("/")[-1][5]]
            data.append([*self.get_feature(wave_form, sr), target])
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
    

        
def emodb_fold_dl(root: str= "data/emodb", fold: int = 5, sr = 16000):
    each_peole = 10 // fold
    leave_out_peole = [[j + i for i in range(each_peole)] for j in range(0, 10 - each_peole + 1, each_peole)]
    return [[emodb_dataset(root, train=True, leave_out_people_id=leave_out_peole[i], sr=sr),emodb_dataset(root, train=False, leave_out_people_id=leave_out_peole[i], sr=sr)] for i in range(fold)], emodb_dataset.emotions

def emodb_fold_ml(root: str= "data/emodb", fold: int = 5, sr = 16000):
    each_peole = 10 // fold
    leave_out_peole = [[j + i for i in range(each_peole)] for j in range(0, 10 - each_peole + 1, each_peole)]
    return [[emodb_dataset(root, train=True, leave_out_people_id=leave_out_peole[i], sr=sr).get_feature_data(),emodb_dataset(root, train=False, leave_out_people_id=leave_out_peole[i], sr=sr).get_feature_data()] for i in range(fold)], emodb_dataset.emotions


        
if __name__ == "__main__":
    # train_dataset = emodb_dataset("data/emodb", train=True, leave_out_people=["03"])
    # test_dataset = emodb_dataset("data/emodb", train=False, leave_out_people=["03"])
    

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

    print(emodb_fold_ml("data/emodb",1)[0])