import wandb
from datasets.savee import savee_fold_dl
from datasets.emodb import emodb_fold_dl
from models.dl import Train_process
import torch
from models.AE import MFCC_AE
from models.CNN import CNN
import warnings
warnings.filterwarnings("ignore")
import argparse

args = argparse.ArgumentParser()
args.add_argument("--epochs", type=int, default=500)
args.add_argument("--model", type=str, default = "ALL")
args = args.parse_args()

clfs = []
if args.model == "CNN":
    clfs.append(CNN)
elif args.model == "AE":
    clfs.append(MFCC_AE)
else:
    clfs = [CNN, MFCC_AE]

savee_fold, savee_labels = savee_fold_dl(fold = 4)
emodb_fold, emodb_labels = emodb_fold_dl(fold = 5)
optimizer = torch.optim.Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for clf in clfs:
    run = wandb.init(project='audio analysis', name=f"ml - {repr(clf)}", reinit=True)
    run.log_code("./")

    try:
        tp = Train_process()
        acc = tp.test_fold(emodb_fold, model_cls=clf, optimizer_cls=optimizer, labels=savee_labels, device=device, run=run, epochs=args.epochs, name="emodb")
        run.log({"emodb - 5": acc})
        
        acc = tp.test_fold(savee_fold, model_cls=clf, optimizer_cls=optimizer, labels=savee_labels, device=device, run=run, epochs=args.epochs, name="savee")
        run.log({"savee - 4": acc})

    finally:
        run.finish()