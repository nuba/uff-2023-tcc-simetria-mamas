import csv
import json


def booleanToSimNao(val):
    return 'Sim' if val else 'Nao'


fread = open('classifier_results.json')
classifier_results = json.load(fread)
fread.close()

header = [
    'Seed',
    'Base',
    'Group',
    'Tratamento',
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
    'AUC\nScore',
    'Haralick',
    'Haralick\nDiff',
    'LBP',
    'LBP\nDiff',
]

values = []

for seed_data in classifier_results['results_per_seeds']:
    for group in seed_data['groupings']:
        for group_classifier_output in group['results_per_classifiers']:
            values.append(
                [
                    seed_data['seed'],
                    group['meta']['base'],
                    group['group_name'],
                    group['meta']['tratamento'],
                    # "\n".join(group['dataset_keys']),
                    len(classifier_results['data']['X']),
                    len(seed_data['partitions']['X_train']),
                    len(seed_data['partitions']['X_test']),
                    group_classifier_output['classifier_name'],
                    group_classifier_output['confusion_matrix']['tp'],
                    group_classifier_output['confusion_matrix']['tn'],
                    group_classifier_output['confusion_matrix']['fp'],
                    group_classifier_output['confusion_matrix']['fn'],
                    '{0:.3f}'.format(group_classifier_output['metrics']['accuracy']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['tpr']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['tnr']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['ppv']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['npv']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['dor']),
                    '{0:.3f}'.format(group_classifier_output['metrics']['auc_score']),
                    booleanToSimNao(group['meta']['uses_haralick']),
                    booleanToSimNao(group['meta']['uses_haralick_differences']),
                    booleanToSimNao(group['meta']['uses_lbp']),
                    booleanToSimNao(group['meta']['uses_lbp_differences'])
                ])

with open('classifier_results.csv', 'w') as file:
    writer = csv.writer(file, dialect='excel')
    writer.writerow(header)
    writer.writerows(values)
