import json
import sys
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import mahotas
import random as rng
from os.path import exists as file_exists
from datetime import datetime

from skimage import feature
from scipy.spatial import distance


def writeExceptionError(error):
    global dataDump, pacienteId, pacienteDataVisita

    fread = open('errors.json')
    errorsDump = json.load(fread)
    fread.close()
    if pacienteId not in errorsDump:
        errorsDump[pacienteId] = {}
    errorsDump[pacienteId][pacienteDataVisita] = {
        "description": error,
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")
    }
    fwrite = open('errors.json', 'w')
    json.dump(errorsDump, fwrite, indent=2)
    fwrite.close()
    print('Done! (see errors.json)', flush=True)


fread = open('blacklist.json')
blacklist = json.load(fread)
fread.close()

matplotlib.interactive(True)
plt.ion()
rng.seed(12345)
haralick_labels = mahotas.features.texture.haralick_labels[:-1]

# healthy or sick?
outputPrefix = sys.argv[1] + '/'
sensorDataFile = sys.argv[2]
# outputPrefix = 'out/healthy/0001/T0001.1.1.S.2012-10-08.00/'
# sensorDataFile = "in/healthy/visual.ic.uff.br/dmi/bancovl/0001/T0001.1.1.S.2012-10-08.00.txt"

pacienteId = outputPrefix.split(os.path.sep)[2]
pacienteLabel = outputPrefix.split(os.path.sep)[1]
pacienteDataVisita = sensorDataFile.split(os.path.sep)[-1].split('.')[-3]

print(
    'Processando paciente {0}, label {1}, visita {2}...'.format(pacienteId, pacienteLabel, pacienteDataVisita),
    end='',
    flush=True)

if pacienteId in blacklist:
    writeExceptionError(
        'Blacklisted ({0}, {1})'.format(blacklist[pacienteId]['reason'], blacklist[pacienteId]['source']))
    sys.exit()

dataDump = {
    "pacienteId": pacienteId,
    "label": pacienteLabel,
    "timestampStart": datetime.now().strftime("%H:%M:%S.%f"),
    "haralick": {},
    "lbp": {
        "distance_labels": ['braycurtis', 'canberra', 'cityblock', 'euclidean', 'jensen_shannon']
    }
}

# sick
# outputPrefix = 'sick/'
# sensorDataFile = "T0138.2.1.S.2013-09-06.00.txt"

os.makedirs(outputPrefix, exist_ok=True)

sensorData = numpy.loadtxt(sensorDataFile)
cv2.imwrite(outputPrefix + '00-raw.png', numpy.uint8(sensorData))

plt.figure()
plt.hist(sensorData.ravel(), bins=256, range=[0, 255])
plt.title("sensorDataRaw")
plt.savefig(outputPrefix + '00_-hist-raw.png')
# plt.show()

sensorDataNormalized = numpy.uint8(
    numpy.interp(
        sensorData,
        [numpy.min(sensorData), numpy.max(sensorData)],
        [0, 255]
    )
)
# cv2.imshow('normalized', sensorDataNormalized)
cv2.imwrite(
    outputPrefix + '00-normalized.png',
    sensorDataNormalized)

plt.figure()
plt.hist(sensorDataNormalized.ravel(), bins=256, range=[0, 255])
plt.title("sensorDataNormalized")
plt.savefig(outputPrefix + '00_-hist-normalized.png')
# plt.show()

sensorDataNormalizedBlurred = cv2.bilateralFilter(sensorDataNormalized, 5, 21, 7)

(T, thresh) = cv2.threshold(
    sensorDataNormalizedBlurred, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# (T, thresh) = cv2.threshold(sensorDataNormalizedBlurred, 80, 255,
#                             cv2.THRESH_BINARY)
# cv2.imshow("Threshold", thresh)
cv2.imwrite(
    outputPrefix + '00-normalized-blurred-threshold-simples.png',
    thresh)

contours, _ = cv2.findContours(
    thresh, mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_NONE)

# Determine center of gravity and orientation using Moments
M = cv2.moments(contours[0])
try:
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
except ZeroDivisionError:
    writeExceptionError("Divisao por zero, nao foi possível calcular o momento.")
    # sys.exit("Divisão por zero, não foi possível calcular o momento.")
    sys.exit()

# Display results
plt.figure()
plt.title("center")
# plt.imshow(thresh, cmap='gray')
plt.scatter(center[0], center[1], marker="X")
plt.scatter(center[0], 0, marker="o")
plt.scatter(center[0], 480, marker="o")
plt.savefig(outputPrefix + "00-normalized-moments.png")


# plt.show()


# parte 2
# já tenho o centro, já tenho a segmentaçao
# 1 recortar uma metade e flip horizontal
#   (tto faz qual, vou escolher a direita) da imagem normalizada

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
    sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
    sharpened = sharpened.round().astype(numpy.uint8)
    if threshold > 0:
        low_contrast_mask = numpy.absolute(image - blurred) < threshold
        numpy.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


sensorDataNormalizedUnsharped = unsharp_mask(sensorDataNormalized)
# cv2.imshow("sensorDataNormalizedUnsharped", sensorDataNormalizedUnsharped)

full_body_masked = cv2.bitwise_and(thresh, sensorDataNormalizedUnsharped)

haralick_features_full_body_masked = mahotas.features.haralick(full_body_masked, return_mean=True, ignore_zeros=True)
# numpy.savetxt(
#     outputPrefix + "00-haralick_features-full-body-masked.csv",
#     [haralick_features_full_body_masked],
#     delimiter=",",
#     fmt='%s')

dataDump["haralick"]["full_body"] = haralick_features_full_body_masked.tolist()

# metade_template = sensorDataNormalized[center[1]:480, 0:640]
metade_esq = sensorDataNormalizedUnsharped[0:480, center[0]:640]
# cv2.imshow("metade template", metade_esq)

# metade_template_flipped = cv2.flip(metade_template, 1)
# cv2.imshow("metade template flipped", metade_template_flipped)

# 2 recortar a metade da mascara e flip horizontal

metade_esq_mask = thresh[0:480, center[0]:640]
# cv2.imshow("metade mask", metade_mask)

metade_esq_masked = cv2.bitwise_and(metade_esq, metade_esq_mask)
# cv2.imshow("metade_template_masked", metade_esq_masked)

metade_dir = sensorDataNormalizedUnsharped[0:480, 0:center[0]]
metade_dir_mask = thresh[0:480, 0:center[0]]
metade_dir_masked = cv2.bitwise_and(metade_dir, metade_dir_mask)
# cv2.imshow("primeira metade_template_masked", metade_dir_masked)

cv2.imwrite(outputPrefix + "03-mama-esq.png", metade_esq)
cv2.imwrite(outputPrefix + "03-mama-esq-masked.png", metade_esq_masked)
cv2.imwrite(outputPrefix + "03-mama-dir.png", metade_dir)
cv2.imwrite(outputPrefix + "03-mama-dir-masked.png", metade_dir_masked)

# NOVO ROTEIRO (Após conversar com a Aura em dom, 6 fev)
# 1. identificar as elipse dos seios
# 2. identificar os mamilo
# 3. dividir os seio em quadrantes
# 4. calcular features de cada quadrante
# 5. apontar as diferenças das features de cada quadrante
# 6. ver sobre distribuição de features por quadrante por dataset (healhy, sick)

# Novo Roteiro, ainda mais enxuto
# gerar estatisticas de cada segmento
# comparar estatitsticas de healthy vs sick
# escrever relatório


haralick_features_metade_esq = mahotas.features.haralick(metade_esq_masked, return_mean=True, ignore_zeros=True)
haralick_features_metade_dir = mahotas.features.haralick(metade_dir_masked, return_mean=True, ignore_zeros=True)
haralick_features_diferenca = numpy.abs(numpy.subtract(haralick_features_metade_esq, haralick_features_metade_dir))
haralick_features_diferenca_squared = numpy.multiply(haralick_features_diferenca, haralick_features_diferenca)

# numpy.savetxt(outputPrefix + "03-haralick_labels.csv", [haralick_labels], delimiter=",", fmt='%s')
#
# numpy.savetxt(
#     "{0}03-haralick_features_mama_esq.csv".format(outputPrefix),
#     [haralick_features_metade_esq],
#     delimiter=",",
#     fmt='%s')
#
# numpy.savetxt(
#     "{0}03-haralick_features_mama_dir.csv".format(outputPrefix),
#     [haralick_features_metade_dir],
#     delimiter=",",
#     fmt='%s')
#
# numpy.savetxt(
#     "{0}03-haralick_features_mama_diff.csv".format(outputPrefix),
#     [haralick_features_diferenca],
#     delimiter=",",
#     fmt='%s')
#
# numpy.savetxt(
#     "{0}03-haralick_features_mama_diff_squared.csv".format(outputPrefix),
#     [haralick_features_diferenca_squared],
#     delimiter=",",
#     fmt='%s')

dataDump["haralick"]["labels"] = haralick_labels
dataDump["haralick"]["corpo_lado_esq"] = haralick_features_metade_esq.tolist()
dataDump["haralick"]["corpo_lado_dir"] = haralick_features_metade_dir.tolist()
dataDump["haralick"]["corpo_lado_diff"] = haralick_features_diferenca.tolist()
dataDump["haralick"]["corpo_lado_diff_squared"] = haralick_features_diferenca_squared.tolist()

# Agora, caso eu tenha feito as máscaras manuais
mask_die_file = outputPrefix + 'die.png'
mask_eie_file = outputPrefix + 'eie.png'
mask_dii_file = outputPrefix + 'dii.png'
mask_eii_file = outputPrefix + 'eii.png'
mask_dse_file = outputPrefix + 'dse.png'
mask_ese_file = outputPrefix + 'ese.png'
mask_dsi_file = outputPrefix + 'dsi.png'
mask_esi_file = outputPrefix + 'esi.png'


def haralick_features_with_quadrant_masks(mask_esq, mask_dir, target_quadrant):
    global metade_dir_masked, metade_esq_masked, dataDump

    metade_esq_quadrant_masked = cv2.bitwise_and(metade_esq_masked, mask_esq)
    metade_dir_quadrant_masked = cv2.bitwise_and(metade_dir_masked, mask_dir)
    metade_dir_quadrant_masked_flipped = cv2.flip(metade_dir_quadrant_masked, 1)
    cv2.imwrite("{0}04-esq-masked-{1}.png".format(outputPrefix, target_quadrant), metade_esq_quadrant_masked)
    cv2.imwrite(
        "{0}04-dir-masked-{1}-flipped.png".format(outputPrefix, target_quadrant),
        metade_dir_quadrant_masked_flipped)

    metade_esq_quadrant_masked_haralick = mahotas.features.haralick(
        metade_esq_quadrant_masked, return_mean=True, ignore_zeros=True)
    metade_dir_quadrant_masked_haralick = mahotas.features.haralick(
        metade_dir_quadrant_masked_flipped, return_mean=True, ignore_zeros=True)

    quadrant_masked_haralick_diferenca = numpy.abs(
        numpy.subtract(metade_esq_quadrant_masked_haralick, metade_dir_quadrant_masked_haralick))
    quadrant_masked_haralick_diferenca_squared = numpy.multiply(
        quadrant_masked_haralick_diferenca,
        quadrant_masked_haralick_diferenca)

    dataDump["haralick"]['quadrante_{0}_esq'.format(target_quadrant)] = metade_esq_quadrant_masked_haralick.tolist()
    dataDump["haralick"]['quadrante_{0}_dir'.format(target_quadrant)] = metade_dir_quadrant_masked_haralick.tolist()
    dataDump["haralick"]['quadrante_{0}_diff'.format(target_quadrant)] = quadrant_masked_haralick_diferenca.tolist()
    dataDump["haralick"][
        'quadrante_{0}_diff_squared'.format(target_quadrant)] = quadrant_masked_haralick_diferenca_squared.tolist()

    # numpy.savetxt(
    #     "{0}04-esq-masked-{1}-haralick_features.csv".format(outputPrefix, target_quadrant),
    #     [metade_esq_quadrant_masked_haralick],
    #     delimiter=",",
    #     fmt='%s')
    # numpy.savetxt(
    #     '{0}04-dir-masked-{1}-haralick_features.csv'.format(outputPrefix, target_quadrant),
    #     [metade_dir_quadrant_masked_haralick],
    #     delimiter=",",
    #     fmt='%s')


def lbp_features_with_quadrant_masks(mask_esq, mask_dir, target_quadrant, dataset_name):
    global metade_dir_masked, metade_esq_masked, dataDump

    radius = 3
    numPoints = 24
    eps = 1e-7

    metade_esq_quadrant_masked = cv2.bitwise_and(metade_esq_masked, mask_esq)
    metade_dir_quadrant_masked = cv2.bitwise_and(metade_dir_masked, mask_dir)
    metade_dir_quadrant_masked_flipped = cv2.flip(metade_dir_quadrant_masked, 1)
    # cv2.imwrite("{0}05-esq-masked-{1}.png".format(outputPrefix, target_quadrant), metade_esq_quadrant_masked)
    # cv2.imwrite("{0}05-dir-masked-{1}-flipped.png".format(outputPrefix, target_quadrant),
    # metade_dir_quadrant_masked_flipped)

    # metade_esq_quadrant_masked_lbp = mahotas.features.lbp(
    #     metade_esq_quadrant_masked, radius, numPoints, ignore_zeros=True)
    # metade_dir_quadrant_masked_lbp = mahotas.features.lbp(
    #     metade_dir_quadrant_masked_flipped, radius, numPoints, ignore_zeros=True)
    #
    # plt.figure()
    # plt.hist(metade_esq_quadrant_masked_lbp)
    # plt.title('metade_esq_quadrant_masked_lbp-{0}'.format(target_quadrant))
    # plt.savefig('{0}05-esq-lbp-hist-{1}.png'.format(outputPrefix, target_quadrant))
    #
    # plt.figure()
    # plt.hist(metade_dir_quadrant_masked_lbp)
    # plt.title('metade_dir_quadrant_masked_lbp-{0}'.format(target_quadrant))
    # plt.savefig('{0}05-dir-lbp-hist-{1}.png'.format(outputPrefix, target_quadrant))
    #
    # numpy.savetxt(
    #     "{0}05-esq-masked-{1}-lbp.csv".format(outputPrefix, target_quadrant),
    #     [metade_esq_quadrant_masked_lbp],
    #     delimiter=",",
    #     fmt='%s')
    # numpy.savetxt(
    #     '{0}05-dir-masked-{1}-lbp.csv'.format(outputPrefix, target_quadrant),
    #     [metade_dir_quadrant_masked_lbp],
    #     delimiter=",",
    #     fmt='%s')

    lbp_esq = feature.local_binary_pattern(metade_esq_quadrant_masked, numPoints, radius, method="uniform")
    (hist_esq, _) = numpy.histogram(lbp_esq.ravel(), bins=numpy.arange(0, numPoints + 3), range=(0, numPoints + 2))
    # normalize the histogram
    hist_esq = hist_esq.astype("float")
    hist_esq /= (hist_esq.sum() + eps)

    lbp_dir = feature.local_binary_pattern(metade_dir_quadrant_masked_flipped, numPoints, radius, method="uniform")
    (hist_dir, _) = numpy.histogram(lbp_dir.ravel(), bins=numpy.arange(0, numPoints + 3), range=(0, numPoints + 2))
    # normalize the histogram
    hist_dir = hist_dir.astype("float")
    hist_dir /= (hist_dir.sum() + eps)

    distances = [
        distance.braycurtis(hist_esq, hist_dir),
        distance.canberra(hist_esq, hist_dir),
        distance.cityblock(hist_esq, hist_dir),
        distance.euclidean(hist_esq, hist_dir),
        distance.jensenshannon(hist_esq, hist_dir, 2.0),
    ]

    # numpy.savetxt(
    #     "{0}05-spatial_distances-values-{1}.csv".format(outputPrefix, target_quadrant),
    #     [distances],
    #     delimiter=",",
    #     fmt='%s')

    # numpy.savetxt(
    #     "{0}05-spatial_distances-labels-{1}.csv".format(outputPrefix, target_quadrant),
    #     [['braycurtis', 'canberra', 'cityblock', 'euclidean', 'jensen_shannon']],
    #     delimiter=",",
    #     fmt='%s')

    # plt.figure()
    # plt.hist(hist_esq)
    # plt.title('metade_esq_quadrant_masked_lbp-{0}'.format(target_quadrant))
    # plt.savefig('{0}05-esq-lbp-hist-{1}.png'.format(outputPrefix, target_quadrant))
    #
    # plt.figure()
    # plt.hist(hist_dir)
    # plt.title('metade_dir_quadrant_masked_lbp-{0}'.format(target_quadrant))
    # plt.savefig('{0}05-dir-lbp-hist-{1}.png'.format(outputPrefix, target_quadrant))

    # numpy.savetxt(
    #     "{0}05-esq-masked-{1}-lbp-via-scikit-image.csv".format(outputPrefix, target_quadrant),
    #     [hist_esq],
    #     delimiter=",",
    #     fmt='%s')

    # numpy.savetxt(
    #     "{0}05-dir-masked-{1}-lbp-via-scikit-image.csv".format(outputPrefix, target_quadrant),
    #     [hist_dir],
    #     delimiter=",",
    #     fmt='%s')

    dataDump["lbp"]['{0}_esq'.format(dataset_name)] = hist_esq.tolist()
    dataDump["lbp"]['{0}_dir'.format(dataset_name)] = hist_dir.tolist()
    dataDump["lbp"]['{0}_distances'.format(dataset_name)] = distances


if (file_exists(mask_dsi_file) and file_exists(mask_esi_file)):
    mask_dsi = cv2.imread(mask_dsi_file, 0)
    mask_esi = cv2.imread(mask_esi_file, 0)
    haralick_features_with_quadrant_masks(mask_esi, mask_dsi, "si")
    lbp_features_with_quadrant_masks(mask_esi, mask_dsi, "si", "quadrante_si")

if (file_exists(mask_die_file) and file_exists(mask_eie_file)):
    mask_die = cv2.imread(mask_die_file, 0)
    mask_eie = cv2.imread(mask_eie_file, 0)
    haralick_features_with_quadrant_masks(mask_eie, mask_die, "ie")
    lbp_features_with_quadrant_masks(mask_eie, mask_die, "ie", "quadrante_ie")

if (file_exists(mask_dii_file) and file_exists(mask_eii_file)):
    mask_dii = cv2.imread(mask_dii_file, 0)
    mask_eii = cv2.imread(mask_eii_file, 0)
    haralick_features_with_quadrant_masks(mask_eii, mask_dii, "ii")
    lbp_features_with_quadrant_masks(mask_eii, mask_dii, "ii", "quadrante_ii")

if (file_exists(mask_dse_file) and file_exists(mask_ese_file)):
    mask_dse = cv2.imread(mask_dse_file, 0)
    mask_ese = cv2.imread(mask_ese_file, 0)
    haralick_features_with_quadrant_masks(mask_ese, mask_dse, "se")
    lbp_features_with_quadrant_masks(mask_ese, mask_dse, "se", "quadrante_se")

if (
        file_exists(mask_dsi_file) and file_exists(mask_esi_file)
        and file_exists(mask_die_file) and file_exists(mask_eie_file)
        and file_exists(mask_dii_file) and file_exists(mask_eii_file)
        and file_exists(mask_dse_file) and file_exists(mask_ese_file)
):
    # tenho como montar uma mascara de todas as mascaras
    mask_dsi = cv2.imread(mask_dsi_file, 0)
    mask_dse = cv2.imread(mask_dse_file, 0)
    mask_dii = cv2.imread(mask_dii_file, 0)
    mask_die = cv2.imread(mask_die_file, 0)

    mask_ds_full = cv2.bitwise_or(mask_dsi, mask_dse)
    mask_di_full = cv2.bitwise_or(mask_dii, mask_die)
    mask_d_full = cv2.bitwise_or(mask_ds_full, mask_di_full)

    mask_esi = cv2.imread(mask_esi_file, 0)
    mask_eie = cv2.imread(mask_eie_file, 0)
    mask_eii = cv2.imread(mask_eii_file, 0)
    mask_ese = cv2.imread(mask_ese_file, 0)

    mask_es_full = cv2.bitwise_or(mask_esi, mask_ese)
    mask_ei_full = cv2.bitwise_or(mask_eii, mask_eie)
    mask_e_full = cv2.bitwise_or(mask_es_full, mask_ei_full)

    haralick_features_with_quadrant_masks(mask_e_full, mask_d_full, "full")
    lbp_features_with_quadrant_masks(mask_e_full, mask_d_full, "full", "mama")

lbp_features_with_quadrant_masks(metade_esq_mask, metade_dir_mask, "corpo_lado", "corpo_lado")

# cv2.waitKey(0)
cv2.destroyAllWindows()


def writeDataDump():
    global dataDump, pacienteId, pacienteDataVisita

    dataDump["timestampEnd"] = datetime.now().strftime("%H:%M:%S.%f")

    fread = open('mydata.json')
    data = json.load(fread)
    fread.close()

    if pacienteId not in data:
        data[pacienteId] = {}

    data[pacienteId][pacienteDataVisita] = dataDump

    fwrite = open('mydata.json', 'w')
    json.dump(data, fwrite, indent=2)
    fwrite.close()
    print('Done!', flush=True)


writeDataDump()
