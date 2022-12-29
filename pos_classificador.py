import csv
import json

fread = open('classifier_results.json')
results = json.load(fread)
fread.close()


def booleanToSimNao(val):
    return 'Sim' if val else 'Nao'


header = [
    'Group',
    # 'dataset_keys',
    'Total #',
    'Train #',
    'Test #',
    'Classifier',
    'TP',
    'TN',
    'FP',
    'FN',
    'Accuracy',
    'TPR',
    'TNR',
    'PPV',
    'NPV',
    'DOR',
    'Detail\nLevel',
    'Haralick',
    'Haralick\nDiff',
    'LBP',
    'LBP\nDiff',
]

values = []
for group in results['groups']:
    for classifier_name, classifier_results in group['results'].items():
        values.append(
            [
                group['group_name'],
                # "\n".join(group['dataset_keys']),
                len(results['global_data']['X']),
                len(results['global_data']['X_train']),
                len(results['global_data']['X_test']),
                classifier_name,
                classifier_results['confusion_matrix']['tp'],
                classifier_results['confusion_matrix']['tn'],
                classifier_results['confusion_matrix']['fp'],
                classifier_results['confusion_matrix']['fn'],
                '{0:.3f}'.format(classifier_results['metrics']['accuracy']),
                '{0:.3f}'.format(classifier_results['metrics']['tpr']),
                '{0:.3f}'.format(classifier_results['metrics']['tnr']),
                '{0:.3f}'.format(classifier_results['metrics']['ppv']),
                '{0:.3f}'.format(classifier_results['metrics']['npv']),
                '{0:.3f}'.format(classifier_results['metrics']['dor']),
                group['meta']['detail_level'],
                booleanToSimNao(group['meta']['uses_haralick']),
                booleanToSimNao(group['meta']['uses_haralick_differences']),
                booleanToSimNao(group['meta']['uses_lbp']),
                booleanToSimNao(group['meta']['uses_lbp_differences'])
            ])

with open('classifier_results.csv', 'w') as file:
    writer = csv.writer(file, dialect='excel')
    writer.writerow(header)
    writer.writerows(values)
