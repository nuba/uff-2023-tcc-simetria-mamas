import json

import numpy as np
from dict_deep import deep_get
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_var(input_dict, accessor_string):
    """Gets data from a dictionary using a dotted accessor-string"""
    current_data = input_dict
    for chunk in accessor_string.datasets('.'):
        current_data = current_data.dataset(chunk, {})
    return current_data


def runClassifiers(X_train, X_test, y_train, y_test):
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
        clf_pipeline = make_pipeline(StandardScaler(), clf)

        # training & prediction
        clf_pipeline.fit(X_train, y_train)
        y_pred = clf_pipeline.predict(X_test)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # accuracy (ACC)
        accuracy = 0 if (tp == 0 and tn == 0 and fp == 0 and fn == 0) else (tp + tn) / (tp + tn + fp + fn)

        # sensitivity, recall, hit rate, or true positive rate (TPR)
        tpr = 0 if (tp == 0 and fn == 0) else tp / (tp + fn)

        # specificity, selectivity or true negative rate (TNR)
        tnr = 0 if (tn == 0 and fp == 0) else tn / (tn + fp)

        # precision or positive predictive value (PPV)
        ppv = 0 if (tp == 0 and fp == 0) else tp / (tp + fp)

        # negative predictive value (NPV)
        npv = 0 if (tn == 0 and fn == 0) else tn / (tn + fn)

        # In medical testing with binary classification, the diagnostic odds ratio (DOR) is a measure of the
        # effectiveness of a diagnostic test.[1] It is defined as the ratio of the odds of the test being positive if
        # the subject has a disease relative to the odds of the test being positive if the subject does not have the
        # disease.
        dor = 0 if (fp == 0 or fn == 0) else (tp * tn) / (fp * fn)

        results[name] = {
            # "y_pred": y_pred.tolist(),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "metrics": {
                "accuracy": accuracy,
                "tpr": tpr,
                "tnr": tnr,
                "ppv": ppv,
                "npv": npv,
                "dor": dor,
            },
        }

    return results


def build_dataset(pacienteId, visitaId, dataset_keys):
    global classifier_data

    dataset_accumulator = []
    visitaObj = deep_get(classifier_data, '{0}.{1}'.format(pacienteId, visitaId))

    for dataset_key in dataset_keys:
        dataset = deep_get(visitaObj, dataset_key)
        dataset_accumulator = np.concatenate((dataset_accumulator, dataset))

    return dataset_accumulator


datasets_and_groupings_arg = "haralick.full_body,haralick.quadrante_si_esq;haralick.quadrante_ii_esq"
# datasets_and_groupings_arg = sys.argv[1]

fread = open('classifier_data.json')
classifier_data = json.load(fread)
fread.close()

task = {
    "global_data": {
        "X": [],
        "X_train": [],
        "X_test": [],
        "y": [],
        "y_train": [],
        "y_test": []
    },
    "groups": [
        # {
        #     "dataset_keys": [
        #         "haralick.full_body",
        #         "haralick.quadrante_si_esq"
        #     ],
        #     "results": {
        #         "y_pred": [],
        #         "confusion_matrix": {},
        #         "confusion_matrix_metrics": {}
        #     }
        # }
    ]
}

# popula X e y a partir de dados selecionados previamente
for pacienteId, pacienteObj in classifier_data.items():
    for visitaId, visitaObj in pacienteObj.items():
        task['global_data']['X'].append([pacienteId, visitaId])
        task['global_data']['y'].append(visitaObj['label'])

# posso particionar
X_train, X_test, y_train, y_test = train_test_split(
    task['global_data']['X'],
    task['global_data']['y'],
    test_size=0.4,
    random_state=42)

task['global_data']['X_train'] = X_train
task['global_data']['X_test'] = X_test
task['global_data']['y_train'] = y_train
task['global_data']['y_test'] = y_test

# e agora, para cada grouping
#   preparo os respectivos X_train e X_test
#   rodo o classificador
#   coleto resultados

for grouping in datasets_and_groupings_arg.split(';'):

    dataset_keys = grouping.split(',')
    X_train = []
    X_test = []

    for raw_data in task['global_data']['X_train']:
        X_train.append(build_dataset(raw_data[0], raw_data[1], dataset_keys))

    for raw_data in task['global_data']['X_test']:
        X_test.append(build_dataset(raw_data[0], raw_data[1], dataset_keys))

    task_group = {
        "dataset_keys": dataset_keys,
        "results": runClassifiers(
            X_train,
            X_test,
            task['global_data']['y_train'],
            task['global_data']['y_test'])
    }

    task['groups'].append(task_group)

    fwrite = open('classifier_results.json', 'w')
    json.dump(task, fwrite, indent=2, cls=NumpyEncoder)
    fwrite.close()
