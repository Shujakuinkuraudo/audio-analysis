from xgboost import XGBClassifier
import torch
from sklearn.metrics import accuracy_score, classification_report
from datasets.emodb import emodb_dataset, emodb_fold_dl
from datasets.savee import savee_dataset, savee_fold_dl
from torch.utils.data import DataLoader
from models.AE import MFCC_AE
import tqdm


class Train_process:
    def __init__(self):
        pass
        
    def test_single(self):
        train_dataset = emodb_dataset("data/emodb", train=True, leave_out_people_id=[0])
        test_dataset = emodb_dataset("data/emodb", train=False, leave_out_people_id=[0])
        
        train_datas, train_targets = train_dataset.get_feature_data()
        test_datas, test_targets = test_dataset.get_feature_data()

        clf = self.clf
        clf.fit(train_datas, train_targets)
        print(accuracy_score(clf.predict(test_datas), test_targets))
        
    def test_fold(self, fold , model_cls:torch.nn.Module,optimizer_cls, labels, device, run=None, epochs=10, name="savee"):
        accs = []
        max_accs = []
        for i, (train_dataset, test_dataset) in (tq := tqdm.tqdm(enumerate(fold), total= len(fold))):
            train_dataloader = DataLoader(train_dataset, batch_size=run.config["batch_size"], shuffle=True, num_workers=20, prefetch_factor=5, persistent_workers=True)
            test_dataloader = DataLoader(test_dataset, batch_size=run.config["batch_size"], shuffle=False, num_workers=20, prefetch_factor=5, persistent_workers=True)
            model = model_cls().to(device)
            optimizer = optimizer_cls(model.parameters(), lr=1e-3)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=run.config["lr_step_size"], gamma=run.config["lr_step_gamma"])
            max_acc = 0

            for epoch in (epoch_tq := tqdm.tqdm(range(epochs))):
                model.train()
                for zcr, energy, mfcc, max_val, fft, mfcc_partial, target in train_dataloader:
                    mfcc, mfcc_partial = mfcc.to(device), mfcc_partial.to(device)
                    target = target.to(device)
                    output = model(mfcc, mfcc_partial)
                    optimizer.zero_grad()
                    loss = model.loss_function(*output, target)
                    loss.backward()
                    optimizer.step()
                if run:
                    run.log({f"{name}_{i}_epoch": epoch, f"{name}_{i}_loss": loss.item()})

                model.eval()
                outputs = []
                targets = []
                for zcr, energy, mfcc, max_val, fft, mfcc_partial, target in test_dataloader:
                    mfcc, mfcc_partial = mfcc.to(device), mfcc_partial.to(device)
                    target = target.to(device)
                    output = model(mfcc, mfcc_partial)[0]
                    outputs.extend(output.argmax(dim=-1).cpu().numpy().tolist())
                    targets.extend(target.cpu().numpy().tolist())
                now_acc = accuracy_score(outputs,targets)
                max_acc = max(now_acc, max_acc)
                if run:
                    run.log({f"{name}_{i}_epoch": epoch, f"{name}_{i}_acc": now_acc, f"{name}_{i}_max_acc": max_acc})
                epoch_tq.set_description(f"Fold {i} epoch {epoch} accuracy: {now_acc} loss: {loss.item()}")

            lr_scheduler.step()
            tq.set_description(f"Fold {i} accuracy: {now_acc}")
            accs.append(now_acc)
            max_accs.append(max_acc)
        return sum(accs) / len(fold), sum(max_accs) / len(fold)

if __name__ == "__main__":
    clf = XGBClassifier()
    tp = Train_process()
    fold, labels = savee_fold_dl(fold = 4)
    tp.test_fold(fold, clf, labels)
    fold, labels = emodb_fold_dl(fold = 5)
    tp.test_fold(fold, clf, labels)


        
    