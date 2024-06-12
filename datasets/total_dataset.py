from torch.utils.data import Dataset, DataLoader
import glob
import torchaudio
import torch
from typing import Tuple, List
from .get_feature import get_feature
import tqdm
class emodb_dataset:
    def __init__(self, emodb_root: str = "data/emodb",savee_root:str = "data/savee", download: bool = True, train=True, leave_out_people_id: List[int] = [], sr = 22050):
        self.sr = sr
        self.time = 5

        self.dataset_people_dict = {}

        self.emodb_dict = {
            "W" :0, # Anger
            "L" :1,
            "E" :2,
            "A" :3,
            "F" :4,
            "T" :5,
            "N" :6
        }
        self.dataset_people_dict["emodb"] = self.get_emodb_data(glob.glob(emodb_root+"/*.wav"))

        self.savee_dict = {
            "a" : 0,
            "d" : 2,
            "f" : 3,
            "h" : 4,
            "n" : 6,
            "sa": 5,
            "su": 1
        }
        self.dataset_people_dict["savee"] = self.get_savee_data(glob.glob(savee_root+"/*.wav"))
    
    def get_emodb_data(self, data_path):
        data_people = {}
        for index in tqdm.tqdm(range(len(data_path))):
            wave_form, sr = torchaudio.load(data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * self.time:
                wave_form = torch.cat(
                    (
                        wave_form,
                        torch.zeros(1, self.sr * self.time-(wave_form.shape[1]//2))
                    )
                    , dim=1)
                wave_form = torch.cat(
                    (
                        torch.zeros(1, self.sr * self.time-(wave_form.shape[1]//2 + (wave_form.shape[1]%2))), wave_form
                        )
                    , dim=1)
            if wave_form.shape[1] > self.sr * self.time:
                pad_len = wave_form.shape[1] - self.sr * self.time
                wave_form = wave_form[:,pad_len//2:pad_len//2 + self.sr * self.time]

            wave_form = wave_form[0]
                
            target = self.emodb_dict[data_path[index].split("/")[-1][5]]
            people = data_path[index].split("/")[-1][:2]

            if people not in data_people:
                data_people[people] = []
            data_people[people].append([get_feature(wave_form, sr), target])
        return data_people

    def get_savee_data(self, data_path):
        data_people = {}
        for index in tqdm.tqdm(range(len(data_path))):
            wave_form, sr = torchaudio.load(data_path[index], format="wav")
            if sr != self.sr:
                wave_form = torchaudio.transforms.Resample(sr, self.sr)(wave_form)
                sr = self.sr
                
            if wave_form.shape[1] < self.sr * self.time:
                wave_form = torch.cat(
                    (
                        wave_form,
                        torch.zeros(1, self.sr * self.time-(wave_form.shape[1]//2))
                    )
                    , dim=1)
                wave_form = torch.cat(
                    (
                        torch.zeros(1, self.sr * self.time-(wave_form.shape[1]//2 + (wave_form.shape[1]%2))), wave_form
                        )
                    , dim=1)
            if wave_form.shape[1] > self.sr * self.time:
                pad_len = wave_form.shape[1] - self.sr * self.time
                wave_form = wave_form[:,pad_len//2:pad_len//2 + self.sr * self.time]

            wave_form = wave_form[0]
                
            target = self.savee_dict[data_path[index].split("/")[-1].split("_")[1][:-6]]
            people = data_path[index].split("/")[-1][:2]


            if people not in data_people:
                data_people[people] = []
            data_people[people].append([get_feature(wave_form, sr), target])
        return data_people

def split_dataset(dataset_people_dict, test_size=0.2, choose="emodb"):
    data = []
    for dataset in dataset_people_dict:
        if dataset == choose:
            for people in dataset_people_dict[dataset]:
                data += dataset_people_dict[dataset][people]
    return data

        
if __name__ == "__main__":
    import wandb
    emodb = emodb_dataset()
    # import numpy as np
    # data = np.load("./SAVEE.npy",allow_pickle=True).item()
    # x_source = data["x"]
    # y_source = data["y"]
    # x_y = list(zip(x_source, y_source))
    # a,b = train_test_split(x_y, test_size=0.2, random_state=42)
    from sklearn.model_selection import train_test_split, KFold
    run = wandb.init(project='audio analysis', name=f"kfold - all - CNN", reinit=True)

    for dataset_name in ["emodb","savee"]:
        data = split_dataset(emodb.dataset_people_dict, choose=dataset_name)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        losses = []
        for i,(train_index, test_index) in enumerate(kf.split(data)):
            a = [data[i] for i in train_index]
            b = [data[i] for i in test_index]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader = DataLoader(a, batch_size=64, shuffle=True)
            test_loader = DataLoader(b, batch_size=16, shuffle=False)
            from models.CNN import CNN
            model = CNN(num_classes=7).to(device)
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.93, 0.98))
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            
            for _ in (tqdm_epoch := tqdm.tqdm(range(1000))):
                model.train()
                losses = []
                for (zcr, energy, mfcc_total, max_val, fft, mfcc_partial), y in (tqdm_train := tqdm.tqdm(train_loader, total=len(train_loader), leave=False)):
                    optimizer.zero_grad()
                    mfcc_total = mfcc_total.to(device)
                    # mfcc_partial = mfcc_partial.to(device)
                    y = y.to(device)
                    # smooth_label
                    y = torch.nn.functional.one_hot(y, num_classes=7).float()
                    y = y * 0.9 + 0.1 / 7
                    
                    # output = model.forward(mfcc_total=mfcc_total, mfcc_partial=mfcc_partial)
                    output = model.forward(mfcc_total=mfcc_total, mfcc_partial=None)
                    loss = model.loss_function(output, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    tqdm_train.set_description(f"loss: {sum(losses)/len(losses)}")
                lr_scheduler.step()
                
                model.eval()
                acc = []
                for (zcr, energy, mfcc_total, max_val, fft, mfcc_partial), y in (tqdm_test := tqdm.tqdm(test_loader, total=len(test_loader), leave=False)):
                    mfcc_total = mfcc_total.to(device)
                    # mfcc_partial = mfcc_partial.to(device)
                    y = y.to(device)

                    output = model.forward(mfcc_total=mfcc_total, mfcc_partial=None)
                    pred = output.argmax(dim=-1)
                    # y = y.argmax(dim=-1)
                    acc.append((pred == y).sum().item() / len(y))
                    tqdm_test.set_description(f"accuracy: {sum(acc)/len(acc)}")
                tqdm_epoch.set_description(f"accuracy: {sum(acc)/len(acc)}")
                run.log({f"{dataset_name}_kfold_{i}_loss": sum(losses)/len(losses), f"{dataset_name}_kfold_{i}_acc": sum(acc)/len(acc), "epoch": _})
            accs.append(sum(acc)/len(acc))
            losses.append(sum(losses)/len(losses))
        run.log({f"{dataset_name}_kfold_acc": sum(accs)/len(accs), f"{dataset_name}_kfold_loss": sum(losses)/len(losses)})



                


        
        