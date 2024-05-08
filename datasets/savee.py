import torch
import tqdm
from torch.utils.data import Dataset
import torchaudio
from typing import List,Literal, Tuple
import glob
class savee_dataset(Dataset):
    emotions = ["a", "d", "f", "h", "n", "sa", "su"]
    def __init__(self, root: str = "data/savee", train=True, leave_out_people_id: List[int] = [], sr = 16000):
        self.sr = sr
        self.emo_dict = {self.emotions[i]:i for i in range(len(self.emotions))}
        self.train = train
        self.people_id = ["DC", "JE", "JK", "KL"]

        self.data_path = self.preprocess(glob.glob(root+"/*.wav"), [self.people_id[i] for i in leave_out_people_id])
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={"n_fft": 400,"win_length":400, "hop_length": 200, "n_mels": 23}, sample_rate=sr)
        
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        wave_form, sr = torchaudio.load(self.data_path[index], format="wav")
        if sr != self.sr:
            wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
            sr = self.sr
            
        if wave_form.shape[1] < 40000:
            wave_form = torch.cat((wave_form, torch.zeros(1, 40000-wave_form.shape[1])), dim=1)
        if wave_form.shape[1] > 40000:
            wave_form = wave_form[:,:40000]

        wave_form = wave_form.mean(dim=0)
        
            
        target = self.emo_dict[self.data_path[index].split("/")[-1].split("_")[1][:-6]]
        return *self.get_feature(wave_form, sr), target
    
    def get_feature(self, wave_form:torch.Tensor, sr) -> Tuple[torch.Tensor]:
        frames = wave_form.unfold(0, 400, 200)
        zcr = frames.sign().diff(dim=1).ne(0).sum(dim=1).float() # 199
        energy = frames.pow(2).sum(dim=1) # 199
        max_val = frames.abs().max(dim=1).values # 199
        fft = torch.fft.rfft(frames, 20).real # 199,11
        
        mfcc = self.mfcc_transform(wave_form) # 13, 201
        return zcr, energy, mfcc, max_val, fft

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
    
    def get_feature_data(self):
        datas = []
        targets = []

        for zcr, energy, mfcc, max_val, fft, target in tqdm.tqdm(self):
            datas.append(torch.cat([zcr, energy, max_val, fft.view(-1)], dim=0))
            targets.append(target)
        return datas, targets

        
def savee_fold_dl(root: str= "data/savee", fold: int = 5, sr = 16000):
    each_peole = 4 // fold
    leave_out_peole = [[j + i for i in range(each_peole)] for j in range(0, 4 - each_peole + 1, each_peole)]
    return [[savee_dataset(root, train=True, leave_out_people_id=leave_out_peole[i], sr=sr),savee_dataset(root, train=False, leave_out_people_id=leave_out_peole[i], sr=sr)] for i in range(fold)]

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