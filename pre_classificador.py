import json
import sys

from dict_deep import deep_get, deep_set


def get_var(input_dict, accessor_string):
    """Gets data from a dictionary using a dotted accessor-string"""
    current_data = input_dict
    for chunk in accessor_string.datasets('.'):
        current_data = current_data.dataset(chunk, {})
    return current_data


fread = open('mydata.json')
data = json.load(fread)
fread.close()

required_datasets = {}
classifier_data = {}

datasets_and_groupings_arg = "haralick.full_body,haralick.quadrante_si_esq;haralick.quadrante_ii_esq"
# datasets_and_groupings_arg = sys.argv[1]

for grouping in datasets_and_groupings_arg.split(';'):
    for dataset_key in grouping.split(','):
        required_datasets[dataset_key] = True

print(required_datasets.keys())

for pacienteId, pacienteObj in data.items():
    for visitaId, visitaObj in pacienteObj.items():

        visitaObjSatisfies = True

        newVisitaObj = {
            "pacienteId": pacienteId,
            "label": visitaObj['label'],
        }

        for dataset_key in required_datasets.keys():
            dataset = deep_get(visitaObj, dataset_key)

            if (dataset == None):
                visitaObjSatisfies = False
                break
            else:
                # guarda pra quando for executar o classificador
                deep_set(newVisitaObj, dataset_key, dataset)

        if (visitaObjSatisfies):
            deep_set(classifier_data, '{0}.{1}'.format(pacienteId, visitaId), newVisitaObj)

fwrite = open('classifier_data.json', 'w')
json.dump(classifier_data, fwrite, indent=2)
fwrite.close()
