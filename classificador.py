import json

import numpy as np
from dict_deep import deep_get
from sklearn import metrics
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


def runClassifiers(X_train, X_test, y_train, y_test):
    # TODO refatorar para usar um factory parametrizado para que os
    #  classifiers possam ser especificados no classifier_plan.json
    classifiers = [
        {
            'name': 'SVC Linear (C=0.0001)',
            'clf': SVC(kernel="linear", C=0.0001)
        },
        {
            'name': 'SVC Linear (C=0.001)',
            'clf': SVC(kernel="linear", C=0.001)
        },
        {
            'name': 'SVC Linear (C=0.01)',
            'clf': SVC(kernel="linear", C=0.01)
        },
        {
            'name': 'SVC Linear (C=0.1)',
            'clf': SVC(kernel="linear", C=0.1)
        },
        {
            'name': 'SVC Linear (C=1)',
            'clf': SVC(kernel="linear", C=1)
        },
        {
            'name': 'SVC Linear (C=10)',
            'clf': SVC(kernel="linear", C=10)
        },
        {
            'name': 'SVC Linear (C=100)',
            'clf': SVC(kernel="linear", C=100)
        },
        {
            'name': 'SVC Linear (C=1,000)',
            'clf': SVC(kernel="linear", C=1000)
        },
        {
            'name': 'SVC Linear (C=10,000)',
            'clf': SVC(kernel="linear", C=10000)
        },
        {
            'name': 'SVC RBF (C=0.0001)',
            'clf': SVC(kernel="rbf", C=0.0001)
        },
        {
            'name': 'SVC RBF (C=0.001)',
            'clf': SVC(kernel="rbf", C=0.001)
        },
        {
            'name': 'SVC RBF (C=0.01)',
            'clf': SVC(kernel="rbf", C=0.01)
        },
        {
            'name': 'SVC RBF (C=0.1)',
            'clf': SVC(kernel="rbf", C=0.1)
        },
        {
            'name': 'SVC RBF (C=1)',
            'clf': SVC(kernel="rbf", C=1)
        },
        {
            'name': 'SVC RBF (C=10)',
            'clf': SVC(kernel="rbf", C=10)
        },
        {
            'name': 'SVC RBF (C=100)',
            'clf': SVC(kernel="rbf", C=100)
        },
        {
            'name': 'SVC RBF (C=1,000)',
            'clf': SVC(kernel="rbf", C=1000)
        },
        {
            'name': 'SVC RBF (C=10,000)',
            'clf': SVC(kernel="rbf", C=10000)
        },
        {
            'name': 'SVC Poly (degree=3, C=0.0001)',
            'clf': SVC(kernel="poly", degree=3, C=0.0001)
        },
        {
            'name': 'SVC Poly (degree=3, C=0.001)',
            'clf': SVC(kernel="poly", degree=3, C=0.001)
        },
        {
            'name': 'SVC Poly (degree=3, C=0.01)',
            'clf': SVC(kernel="poly", degree=3, C=0.01)
        },
        {
            'name': 'SVC Poly (degree=3, C=0.1)',
            'clf': SVC(kernel="poly", degree=3, C=0.1)
        },
        {
            'name': 'SVC Poly (degree=3, C=1)',
            'clf': SVC(kernel="poly", degree=3, C=1)
        },
        {
            'name': 'SVC Poly (degree=3, C=10)',
            'clf': SVC(kernel="poly", degree=3, C=10)
        },
        {
            'name': 'SVC Poly (degree=3, C=100)',
            'clf': SVC(kernel="poly", degree=3, C=100)
        },
        {
            'name': 'SVC Poly (degree=3, C=1,000)',
            'clf': SVC(kernel="poly", degree=3, C=1000)
        },
        {
            'name': 'SVC Poly (degree=3, C=10,000)',
            'clf': SVC(kernel="poly", degree=3, C=10000)
        },
        {
            'name': 'SVC Poly (degree=5, C=0.0001)',
            'clf': SVC(kernel="poly", degree=5, C=0.0001)
        },
        {
            'name': 'SVC Poly (degree=5, C=0.001)',
            'clf': SVC(kernel="poly", degree=5, C=0.001)
        },
        {
            'name': 'SVC Poly (degree=5, C=0.01)',
            'clf': SVC(kernel="poly", degree=5, C=0.01)
        },
        {
            'name': 'SVC Poly (degree=5, C=0.1)',
            'clf': SVC(kernel="poly", degree=5, C=0.1)
        },
        {
            'name': 'SVC Poly (degree=5, C=1)',
            'clf': SVC(kernel="poly", degree=5, C=1)
        },
        {
            'name': 'SVC Poly (degree=5, C=10)',
            'clf': SVC(kernel="poly", degree=5, C=10)
        },
        {
            'name': 'SVC Poly (degree=5, C=100)',
            'clf': SVC(kernel="poly", degree=5, C=100)
        },
        {
            'name': 'SVC Poly (degree=5, C=1,000)',
            'clf': SVC(kernel="poly", degree=5, C=1000)
        },
        {
            'name': 'SVC Poly (degree=5, C=10,000)',
            'clf': SVC(kernel="poly", degree=5, C=10000)
        },
        {
            'name': 'Nearest Neighbors (k=2)',
            'clf': KNeighborsClassifier(2)
        },
        {
            'name': 'Nearest Neighbors (k=5)',
            'clf': KNeighborsClassifier(5)
        },
        {
            'name': 'Nearest Neighbors (k=8)',
            'clf': KNeighborsClassifier(8)
        },
    ]
    results_per_classifiers = []

    classifier_counter = 1

    for classifier in classifiers:
        classifier['name'] = '{0:02d} {1}'.format(classifier_counter, classifier['name'])
        classifier_counter = classifier_counter + 1
        clf_pipeline = make_pipeline(StandardScaler(), classifier['clf'])

        # training & prediction
        clf_pipeline.fit(X_train, y_train)
        y_pred = clf_pipeline.predict(X_test)
        print('.', end='', flush=True)

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

        sm_fpr, sm_tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc_score = metrics.auc(sm_fpr, sm_tpr)

        results_per_classifiers.append(
            {
                'classifier_name': classifier['name'],
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
                    "roc_fpr": sm_fpr,
                    "roc_tpr": sm_tpr,
                    "auc_score": auc_score
                },
            }
        )

    return results_per_classifiers


def build_dataset(pacienteId, visitaId, dataset_keys, viable_data):
    dataset_accumulator = []
    visitaObj = deep_get(viable_data, '{0}.{1}'.format(pacienteId, visitaId))

    for dataset_key in dataset_keys:
        dataset = deep_get(visitaObj, dataset_key)
        dataset_accumulator = np.concatenate((dataset_accumulator, dataset))

    return dataset_accumulator


def build_data(viable_data):
    # popula X e y a partir de dados selecionados previamente
    X = []
    y = []

    for pacienteId, pacienteObj in viable_data.items():
        for visitaId, visitaObj in pacienteObj.items():
            X.append([pacienteId, visitaId])
            y.append(1 if visitaObj['label'] == 'sick' else 0)

    return {'X': X, 'y': y}


def classificationTaskRunner(X, y, viable_data, task_plan, splitting_test_size, splitting_seed):
    task = {
        'seed': splitting_seed,
        'split': splitting_test_size,
        'partitions': {
            'X_train': [],
            'X_test': [],
            'y_train': [],
            'y_test': []
        },
        'groupings': []
    }

    # posso particionar
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=splitting_test_size,
        random_state=splitting_seed)
    task['partitions']['X_train'] = X_train
    task['partitions']['X_test'] = X_test
    task['partitions']['y_train'] = y_train
    task['partitions']['y_test'] = y_test

    # e agora, para cada grouping
    #   preparo os respectivos X_train e X_test
    #   rodo o classificador
    #   coleto resultados
    for grouping in task_plan:

        dataset_keys = grouping['dataset_keys']
        X_train = []
        X_test = []

        for raw_data in task['partitions']['X_train']:
            X_train.append(build_dataset(raw_data[0], raw_data[1], dataset_keys, viable_data))

        for raw_data in task['partitions']['X_test']:
            X_test.append(build_dataset(raw_data[0], raw_data[1], dataset_keys, viable_data))

        print('{0}:'.format(grouping['name']), end='', flush=True)
        task['groupings'].append(
            {
                "group_name": grouping['name'],
                "meta": grouping['meta'],
                "dataset_keys": dataset_keys,
                "results_per_classifiers": runClassifiers(
                    X_train,
                    X_test,
                    task['partitions']['y_train'],
                    task['partitions']['y_test'])
            })

    return task


fread = open('classifier_data.json')
classifier_data = json.load(fread)
fread.close()

fread = open('classifier_plan.json')
classifier_plan = json.load(fread)
fread.close()

data = build_data(classifier_data)

run_count = 30
# splitting_sizes = [0.4, 0.3, 0.2]
splitting_sizes = [0.3]
results_for_seed = []

for seed in np.arange(1, run_count + 1, 1):
    for splitting_size in splitting_sizes:
        print(
            'Running for seed {0}/{1} and splitting size {2}: '.format(
                seed, run_count, splitting_size), end='', flush=True)
        results_for_seed.append(
            classificationTaskRunner(
                data['X'],
                data['y'],
                classifier_data,
                classifier_plan,
                splitting_size,
                seed)
        )
        print(' and done!', flush=True)

fwrite = open('classifier_results.json', 'w')
json.dump(
    {
        'data': data,
        'results_per_seeds': results_for_seed
    }, fwrite, indent=2, cls=NumpyEncoder)
fwrite.close()
