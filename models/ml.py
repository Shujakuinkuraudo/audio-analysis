from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from datasets.emodb import emodb_dataset, emodb_fold_ml
from datasets.savee import savee_dataset, savee_fold_ml
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
        
    def test_fold(self, fold , clf, labels):
        acc = 0
        for i, (train, test) in (tq := tqdm.tqdm(enumerate(fold), total= len(fold))):
            train_datas, train_targets = train
            test_datas, test_targets = test
            clf.fit(train_datas, train_targets)
            now_acc = accuracy_score(clf.predict(test_datas), test_targets)
            acc += now_acc
            print(classification_report(clf.predict(test_datas), test_targets, target_names=labels))
            tq.set_description(f"Fold {i} accuracy: {now_acc}")
        return acc / len(fold)

if __name__ == "__main__":
    clf = XGBClassifier()
    tp = Train_process()
    fold, labels = savee_fold_ml(fold = 4)
    tp.test_fold(fold, clf, labels)
    fold, labels = emodb_fold_ml(fold = 5)
    tp.test_fold(fold, clf, labels)


        
    