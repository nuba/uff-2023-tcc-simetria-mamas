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


def runClassifiers(X_train, X_test, y_train, y_test):
    classifier_names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
    ]

    # TODO refatorar para usar um factory parametrizado para que os
    #  classifiers possam ser especificados no classifier_plan.json
    classifiers = [
        KNeighborsClassifier(2),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1)
    ]
    results_per_classifiers = []

    for classifier_name, clf in zip(classifier_names, classifiers):
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

        results_per_classifiers.append(
            {
                'classifier_name': classifier_name,
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

run_count = 20
test_size = 0.4
results_for_seed = []



for seed in np.arange(1, run_count+1, 1):
    print('Running for seed {0}/{1}'.format(seed, run_count))
    results_for_seed.append(
        classificationTaskRunner(
            data['X'],
            data['y'],
            classifier_data,
            classifier_plan,
            test_size,
            seed)
    )

fwrite = open('classifier_results.json', 'w')
json.dump(
    {
        'data': data,
        'test_size': test_size,
        'results_per_seeds': results_for_seed
    }, fwrite, indent=2, cls=NumpyEncoder)
fwrite.close()
