
import wandb
from datasets.savee import savee_fold_ml
from datasets.emodb import emodb_fold_ml
from models.ml import Train_process

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


clfs = [SVC(C=0.1,gamma="auto"), XGBClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier()]

for clf in clfs:
    run = wandb.init(project='audio analysis', name=f"kfold - all - {repr(clf)}", reinit=True)
    feature_cache = {}

    try:
        tp = Train_process()
        emodb_fold, emodb_labels = emodb_fold_ml(fold = 5, feature_cache= feature_cache)
        acc,accs = tp.test_fold(emodb_fold, clf, emodb_labels)
        run.log({"emodb - 5 - acc": acc})
        for i in range(5):
            run.log({f"emodb_{i}_acc": accs[i]})

        savee_fold, savee_labels = savee_fold_ml(fold = 4, feature_cache= feature_cache)
        acc,accs = tp.test_fold(savee_fold, clf, savee_labels)
        run.log({"savee - 4 - acc": acc})
        for i in range(4):
            run.log({f"savee_{i}_acc": accs[i]})

    finally:
        run.finish()