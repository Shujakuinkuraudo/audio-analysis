
import wandb
import xgboost
from xgboost import XGBClassifier
from datasets.savee import savee_fold_ml
from datasets.emodb import emodb_fold_ml
from models.ml import Train_process

clf = XGBClassifier()
run = wandb.init(project='audio analysis', name=f"ml - {repr(clf)}")
run.log_code(".")


tp = Train_process()
fold, labels = savee_fold_ml(fold = 4)
acc = tp.test_fold(fold, clf, labels)
run.log({"savee - 4": acc})

fold, labels = emodb_fold_ml(fold = 5)
tp.test_fold(fold, clf, labels)
run.log({"emodb - 5": acc})