from torch.utils.data import Dataset, DataLoader
import glob
import torchaudio
import torch
from typing import Tuple, List
from .get_feature import get_feature
import tqdm
class emodb_dataset:
    def __init__(self, emodb_root: str = "data/emodb",savee_root:str = "data/savee", download: bool = True, train=True, leave_out_people_id: List[int] = [], sr = 16000):
        self.sr = sr
        self.time = 4
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=13, melkwargs={"n_fft": 800,"win_length":200, "hop_length": 50, "n_mels": 23}, sample_rate=sr)

        self.emodb_dict = {
            "W" :0, # Anger
            "L" :1,
            "E" :2,
            "A" :3,
            "F" :4,
            "T" :5,
            "N" :6
        }
        self.dataset_people_dict = {}
        self.dataset_people_dict["emodb"] = self.get_emodb_data(glob.glob(emodb_root+"/*.wav"))

        # self.savee_dict = {
        #     "a" : 0,
        #     "d" : 2,
        #     "f" : 3,
        #     "h" : 4,
        #     "n" : 6,
        #     "sa": 5,
        #     "su": 7
        # }
        # self.dataset_people_dict["savee"] = self.get_savee_data(glob.glob(savee_root+"/*.wav"))
    
    def get_emodb_data(self, data_path):
        data_people = {}
        for index in tqdm.tqdm(range(len(data_path))):
            wave_form, sr = torchaudio.load(data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * self.time:
                wave_form = torch.cat((wave_form, torch.zeros(1, self.sr * self.time-wave_form.shape[1])), dim=1)
            if wave_form.shape[1] > self.sr * self.time:
                wave_form = wave_form[:,:self.sr * self.time]

            wave_form = wave_form.mean(dim=0)
                
            target = self.emodb_dict[data_path[index].split("/")[-1][5]]
            people = data_path[index].split("/")[-1][:2]

            if people not in data_people:
                data_people[people] = []
            else:
                data_people[people].append([get_feature(wave_form, sr, mfcc_transform=self.mfcc_transform), target])
        return data_people

    def get_savee_data(self, data_path):
        data_people = {}
        for index in tqdm.tqdm(range(len(data_path))):
            wave_form, sr = torchaudio.load(data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * self.time:
                wave_form = torch.cat((wave_form, torch.zeros(1, self.sr * self.time-wave_form.shape[1])), dim=1)
            if wave_form.shape[1] > self.sr * self.time:
                wave_form = wave_form[:,:self.sr * self.time]

            wave_form = wave_form.mean(dim=0)
                
            target = self.savee_dict[data_path[index].split("/")[-1].split("_")[1][:-6]]
            people = data_path[index].split("/")[-1][:2]


            if people not in data_people:
                data_people[people] = []
            else:
                data_people[people].append([get_feature(wave_form, sr, mfcc_transform=self.mfcc_transform), target])
        return data_people

from sklearn.model_selection import train_test_split
def split_dataset(dataset_people_dict, test_size=0.2):
    data = []
    for dataset in dataset_people_dict:
        for people in dataset_people_dict[dataset]:
            data += dataset_people_dict[dataset][people]
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data

        
if __name__ == "__main__":
    emodb = emodb_dataset()
    a,b = split_dataset(emodb.dataset_people_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(a, batch_size=16, shuffle=True)
    test_loader = DataLoader(b, batch_size=32, shuffle=False)
    print(len(a), len(b))
    from models.CNN import CNN
    model = CNN(num_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for _ in tqdm.tqdm(range(100)):
        losses = []
        for (zcr, energy, mfcc_total, max_val, fft, mfcc_partial), y in (tqdm_train := tqdm.tqdm(train_loader, total=len(train_loader), leave=False)):
            optimizer.zero_grad()
            mfcc_total = mfcc_total.to(device)
            mfcc_partial = mfcc_partial.to(device)
            y = y.to(device)
            
            output = model.forward(mfcc_total=mfcc_total, mfcc_partial=mfcc_partial)
            loss = model.loss_function(output, y)
            loss.backward()
            losses.append(loss.item())
            tqdm_train.set_description(f"loss: {sum(losses)/len(losses)}")
        
        for (zcr, energy, mfcc_total, max_val, fft, mfcc_partial), y in (tqdm_test := tqdm.tqdm(test_loader, total=len(test_loader), leave=False)):
            mfcc_total = mfcc_total.to(device)
            mfcc_partial = mfcc_partial.to(device)
            y = y.to(device)

            output = model.forward(mfcc_total=mfcc_total, mfcc_partial=mfcc_partial)
            pred = output.argmax(dim=-1)
            tqdm_test.set_description(f"accuracy: {(pred == y).sum().item() / len(y)}")


            


    
    