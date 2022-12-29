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

fread = open('classifier_plan.json')
classifier_plan = json.load(fread)
fread.close()

fread = open('blacklist.json')
blacklist = json.load(fread)
fread.close()

required_datasets = {}
classifier_data = {}
classifier_data_rejects = []

for grouping in classifier_plan:
    for dataset_key in grouping['dataset_keys']:
        required_datasets[dataset_key] = True

# print(required_datasets.keys())

for pacienteId, pacienteObj in data.items():
    for visitaId, visitaObj in pacienteObj.items():

        visitaObjSatisfies = True
        rejection_reasons = []

        newVisitaObj = {
            "pacienteId": pacienteId,
            "label": visitaObj['label'],
        }
        if pacienteId in blacklist:
            visitaObjSatisfies = False
            rejection_reasons.append('Blacklisted ({0}, {1})'.format(blacklist[pacienteId]['reason'], blacklist[pacienteId]['source']))
        else:

            for dataset_key in required_datasets.keys():
                dataset = deep_get(visitaObj, dataset_key)

                if (dataset == None):
                    rejection_reasons.append('Missing {0}'.format(dataset_key))
                    visitaObjSatisfies = False
                else:
                    # guarda pra quando for executar o classificador
                    deep_set(newVisitaObj, dataset_key, dataset)

        if (visitaObjSatisfies):
            deep_set(classifier_data, '{0}.{1}'.format(pacienteId, visitaId), newVisitaObj)
        else:
            classifier_data_rejects.append(
                {
                    'pacienteId': pacienteId,
                    'visitaID': visitaId,
                    'label': visitaObj['label'],
                    'reasons': rejection_reasons
                })
fwrite = open('classifier_data.json', 'w')
json.dump(classifier_data, fwrite, indent=2)
fwrite.close()

fwrite = open('classifier_data_rejects.json', 'w')
json.dump(classifier_data_rejects, fwrite, indent=2)
fwrite.close()
