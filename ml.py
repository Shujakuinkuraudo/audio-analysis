import wandb
from datasets.savee import savee_fold_ml
from datasets.emodb import emodb_fold_ml
from models.ml import Train_process

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


clfs = [
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    SVC(),
    RandomForestClassifier(),
    XGBClassifier(),
]
savee_fold, savee_labels = savee_fold_ml(fold=4)
emodb_fold, emodb_labels = emodb_fold_ml(fold=5)

config = {
    "batch_size": 80,
}

for clf in clfs:
    run = wandb.init(
        project="audio analysis", name=f"ml - {repr(clf)}", reinit=True, config=config
    )

    try:
        tp = Train_process()
        acc = tp.test_fold(savee_fold, clf, savee_labels)
        run.log({"savee - 4": acc})

        acc = tp.test_fold(emodb_fold, clf, emodb_labels)
        run.log({"emodb - 5": acc})
    finally:
        run.finish()
