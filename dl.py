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
args.add_argument("--epochs", type=int, default=1000)
args.add_argument("--model", type=str, default = "ALL")
args = args.parse_args()

clfs = []
if args.model == "CNN":
    clfs.append(CNN)
elif args.model == "AE":
    clfs.append(MFCC_AE)
else:
    clfs = [CNN, MFCC_AE]

config = {
    "batch_size": 80,
    "lr_step_size" : 50,
    "lr_step_gamma" : 0.7,
    "optimizer" : "torch.optim.Adam",
    "win_length" : 512,
    "hop_length" : 256,
    "n_fft" : 2048
}

savee_fold, savee_labels = savee_fold_dl(fold = 4, n_fft=config["n_fft"], win_length=config["win_length"], hop_length=config["hop_length"])
emodb_fold, emodb_labels = emodb_fold_dl(fold = 5, n_fft=config["n_fft"], win_length=config["win_length"], hop_length=config["hop_length"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for clf in clfs:
    run = wandb.init(project='audio analysis', name=f"ml - {repr(clf)}", reinit=True, config = config)

    optimizer = eval(run.config["optimizer"])

    try:
        tp = Train_process()
        acc,max_acc = tp.test_fold(savee_fold, model_cls=clf, optimizer_cls=optimizer, labels=savee_labels, device=device, run=run, epochs=args.epochs, name="savee")
        run.log({"savee - 4 - acc": acc, "savee - 4 - maxacc": max_acc})

        acc,max_acc = tp.test_fold(emodb_fold, model_cls=clf, optimizer_cls=optimizer, labels=savee_labels, device=device, run=run, epochs=args.epochs, name="emodb")
        run.log({"emodb - 5 - acc": acc, "emodb - 5 - maxacc": max_acc})
        

    finally:
        run.finish()