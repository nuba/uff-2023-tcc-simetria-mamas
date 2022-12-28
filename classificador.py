import json
import sys
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from datetime import datetime

from dict_deep import deep_get


def get_var(input_dict, accessor_string):
    """Gets data from a dictionary using a dotted accessor-string"""
    current_data = input_dict
    for chunk in accessor_string.datasets('.'):
        current_data = current_data.dataset(chunk, {})
    return current_data


fread = open('mydata.json')
data = json.load(fread)
fread.close()

# datasets_arg = "haralick.full_body,haralick.quadrante_si_esq"
datasets_arg = sys.argv[1]
datasets = datasets_arg.split(',')

X_rich = []
X = []
y = []

for pacienteId, pacienteObj in data.items():
    for visitaId, visitaObj in pacienteObj.items():

        visitaObjSatisfies = True
        dataset_accumulator = []

        for dataset_key in datasets:
            # check if paciente e visita satisfazem os datasets
            # caso positivo, empurra a tupla para o X, Y
            dataset = deep_get(visitaObj, dataset_key)
            if (dataset != None):
                dataset_accumulator = np.concatenate((dataset_accumulator, dataset))
            else:
                visitaObjSatisfies = False
                break

        if (visitaObjSatisfies):
            X_rich.append([pacienteId, visitaId, dataset_accumulator.tolist()])
            y.append(visitaObj['label'])

# agora eu tenho um X_rich e um y que posso particionar

X_rich_train, X_rich_test, y_train, y_test = train_test_split(X_rich, y, test_size=0.4, random_state=42)

X_train_info = []
X_train = []
X_test_info = []
X_test = []

for raw_data in X_rich_train:
    X_train_info.append([raw_data[0], raw_data[1]])
    X_train.append(raw_data[2])

for raw_data in X_rich_test:
    X_test_info.append([raw_data[0], raw_data[1]])
    X_test.append(raw_data[2])

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1)
]
results = {}

for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    timestampTrainingStart = datetime.now().strftime("%H:%M:%S.%f"),
    clf.fit(X_train, y_train)
    timestampTrainingEnd = datetime.now().strftime("%H:%M:%S.%f"),
    score = clf.score(X_test, y_test)
    timestampClassifyingStart = datetime.now().strftime("%H:%M:%S.%f"),
    y_pred = clf.predict(X_test)
    timestampClassifyingEnd = datetime.now().strftime("%H:%M:%S.%f"),
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # accuracy (ACC)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    tpr = tp / (tp + fn)

    # specificity, selectivity or true negative rate (TNR)
    tnr = tn / (tn + fp)

    # precision or positive predictive value (PPV)
    ppv = tp / (tp + fp)

    # negative predictive value (NPV)
    npv = 0 if (tn == 0 and fn == 0) else tn / (tn + fn)

    # In medical testing with binary classification, the diagnostic odds ratio (DOR) is a measure of the
    # effectiveness of a diagnostic test.[1] It is defined as the ratio of the odds of the test being positive if the
    # subject has a disease relative to the odds of the test being positive if the subject does not have the disease.
    # dor = (tp * tn) / (fp * fn)

    results[name] = {
        "score": score,
        "y_pred": y_pred.tolist(),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "confusion_matrix_metrics": {
            "accuracy": accuracy,
            "tpr": tpr,
            "tnr": tnr,
            "ppv": ppv,
            "npv": npv,
            # "dor": dor,
        },
        "times": {
            "timestampTrainingStart": timestampTrainingStart,
            "timestampTrainingEnd": timestampTrainingEnd,
            "timestampClassifyingStart": timestampClassifyingStart,
            "timestampClassifyingEnd": timestampClassifyingEnd
        }
    }

    data = {
        "X_rich": X_rich,
        "X_rich_train": X_rich,
        "X_rich_test": X_rich,
        "y": y,
        "X_train": X_train,
        "X_train_info": X_train_info,
        "X_test": X_test,
        "X_test_info": X_test_info,
        "y_train": y_train,
        "y_test": y_test
    }

    fwrite = open('classifier_selected_data.json', 'w')
    json.dump(
        {
            "datasets": datasets,
            "data": data,
            "results": results
        }, fwrite, indent=2)
    fwrite.close()
