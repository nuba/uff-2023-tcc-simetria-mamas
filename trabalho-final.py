import sys
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import mahotas
from matplotlib.colors import NoNorm
import random as rng


matplotlib.interactive(True)
plt.ion()
rng.seed(12345)


# healthy or sick?
outputPrefix = sys.argv[1] + '/'
sensorDataFile = sys.argv[2]

os.makedirs(outputPrefix, exist_ok=True)

# outputPrefix = 'healthy/'
#sensorDataFile = "T0001.1.1.S.2012-10-08.00.txt"

# sick
# outputPrefix = 'sick/'
# sensorDataFile = "T0138.2.1.S.2013-09-06.00.txt"

sensorData = numpy.loadtxt(sensorDataFile)
# cv2.imshow('raw', numpy.uint8(sensorData))
cv2.imwrite(outputPrefix + '00-raw.png', numpy.uint8(sensorData))

plt.figure()
plt.hist(sensorData.ravel(), bins=256, range=[0, 255])
plt.title("sensorDataRaw")
plt.savefig(outputPrefix + '00_-hist-raw.png')
plt.show()

sensorDataNormalized = numpy.uint8(
  numpy.interp(
    sensorData,
    [numpy.min(sensorData), numpy.max(sensorData)],
    [0, 255]
  )
)
cv2.imshow('normalized', sensorDataNormalized)
cv2.imwrite(outputPrefix + '00-normalized.png',
            sensorDataNormalized)

plt.figure()
plt.hist(sensorDataNormalized.ravel(), bins=256, range=[0, 255])
plt.title("sensorDataNormalized")
plt.savefig(outputPrefix + '00_-hist-normalized.png')
plt.show()


# sensorDataHistogramEqualized = cv2.equalizeHist(sensorDataNormalized)
# cv2.imshow('normalized-histogram-equalized', sensorDataNormalized)
# cv2.imwrite(outputPrefix + '00-normalized-histogram-equalized.png',
#             sensorDataNormalized)
#
# plt.figure()
# plt.hist(sensorDataHistogramEqualized.ravel(), bins=256, range=[0, 255])
# plt.title("sensorDataHistogramEqualized")
# plt.savefig(outputPrefix + '00_-hist-normalized-histogram-equalized.png')
# plt.show()
#
# claheAfterNormalized = cv2.createCLAHE()
# sensorDataNormalizedClahe = claheAfterNormalized.apply(
#   sensorDataNormalized) + 30

# clahefromRaw = cv2.createCLAHE()
# sensorDataRawClahe = clahefromRaw.apply(
#   numpy.uint8(sensorData)
# ) + 30
#
# plt.figure()
# cv2.imshow('normalized-clahe', sensorDataNormalizedClahe)
# cv2.imwrite(outputPrefix + '00-normalized-clahe.png', sensorDataNormalizedClahe)
# plt.hist(sensorDataNormalizedClahe.ravel(), bins=256, range=[0, 255])
# plt.title("sensorDataNormalizedClahe")
# plt.savefig(outputPrefix + '00_-hist-normalized-clahe.png')
# plt.show()
#
# plt.figure()
# cv2.imshow('raw-clahe', sensorDataRawClahe)
# cv2.imwrite(outputPrefix + '00-raw-clahe.png', sensorDataRawClahe)
# plt.hist(sensorDataRawClahe.ravel(), bins=256, range=[0, 255])
# plt.title("sensorDataRawClahe")
# plt.savefig(outputPrefix + '00_-hist-raw-clahe.png')
# plt.show()

# tentar um colormap a partir do domínio da frequência
# sensorDataNormalized

def split_to_rgb_on_frequency():
  rows, cols = sensorDataNormalized.shape
  f = numpy.fft.fft2(sensorDataNormalized)
  fshift = numpy.fft.fftshift(f)
  magnitude_spectrum = 20 * numpy.log(numpy.abs(fshift))

  plt.subplot(121), plt.imshow(sensorDataNormalized, cmap='gray', norm=NoNorm())
  plt.title('Input Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
  plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
  plt.show()

  crow, ccol = rows / 2, cols / 2

  fshift[int(crow - 30):int(crow + 30), int(ccol - 30):int(ccol + 30)] = 0
  f_ishift = numpy.fft.ifftshift(fshift)
  sensorDataNormalized_back = numpy.fft.ifft2(f_ishift)
  sensorDataNormalized_back = numpy.abs(sensorDataNormalized_back)

  innerRadius, outerRadius = 20, 40

  def on_trackbar_outer_radius(val):
    global outerRadius
    outerRadius = val
    print(
      'innerRadius, outerRadius: ' + str(innerRadius) + ', ' + str(outerRadius))
    on_trackbar_outer_and_inner_radius_handler(innerRadius, outerRadius)

  def on_trackbar_inner_radius(val):
    global innerRadius
    innerRadius = val
    print(
      'innerRadius, outerRadius: ' + str(innerRadius) + ', ' + str(outerRadius))
    on_trackbar_outer_and_inner_radius_handler(innerRadius, outerRadius)

  def on_trackbar_outer_and_inner_radius_handler(innerRadius, outerRadius):
    f_highPass = numpy.fft.fft2(sensorDataNormalized)
    mask_highPass = numpy.fft.fftshift(f_highPass)
    mask_highPass[
    int(crow - outerRadius):int(crow + outerRadius),
    int(ccol - outerRadius):int(ccol + outerRadius)] = 0

    channel_highPass = numpy.fft.ifft2(mask_highPass)
    channel_highPass = numpy.abs(channel_highPass)

    f_lowPass = numpy.fft.fft2(sensorDataNormalized)
    mask_lowPass = numpy.fft.fftshift(f_lowPass)
    mask_lowPass[0:int(crow - innerRadius), 0:cols] = 0
    mask_lowPass[int(crow + innerRadius):rows, 0:cols] = 0
    mask_lowPass[0:rows, 0:int(ccol - innerRadius)] = 0
    mask_lowPass[0:rows, int(ccol + innerRadius):cols] = 0

    channel_lowPass = numpy.fft.ifft2(mask_lowPass)
    channel_lowPass = numpy.abs(channel_lowPass)

    f_middlePass = numpy.fft.fft2(sensorDataNormalized)
    mask_middlePass = numpy.fft.fftshift(f_middlePass)
    mask_middlePass[0:int(crow - outerRadius), 0:cols] = 0
    mask_middlePass[int(crow + outerRadius):rows, 0:cols] = 0
    mask_middlePass[0:rows, 0:int(ccol - outerRadius)] = 0
    mask_middlePass[0:rows, int(ccol + outerRadius):cols] = 0
    mask_middlePass[
    int(crow - innerRadius):int(crow + innerRadius),
    int(ccol - innerRadius):int(ccol + innerRadius)
    ] = 0

    channel_middlePass = numpy.fft.ifft2(mask_middlePass)
    channel_middlePass = numpy.abs(channel_middlePass)

    sensorDataSplitOnFrequencyDomain = cv2.merge(
      (channel_lowPass, channel_middlePass, channel_highPass))
    cv2.imshow("sensorDataSplitOnFrequencyDomain",
               sensorDataSplitOnFrequencyDomain)

    cv2.imshow("channel_lowPass", channel_lowPass)
    cv2.imshow("channel_middlePass", channel_middlePass)
    cv2.imshow("channel_highPass", channel_highPass)
    cv2.imwrite(outputPrefix + '00-split-on-frequency.png',
                sensorDataSplitOnFrequencyDomain)

  cv2.namedWindow("controls")

  cv2.createTrackbar("innerRadius", "controls", 0, 240,
                     on_trackbar_inner_radius)
  cv2.createTrackbar("outerRadius", "controls", 0, 240,
                     on_trackbar_outer_radius)
  # Show some stuff

  on_trackbar_outer_and_inner_radius_handler(innerRadius, outerRadius)


def sensorDataNormalizedToMultipleColorMaps():
  print("duh")
  # sensorData_COLORMAP_AUTUMN = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_AUTUMN)
  # cv2.imshow('COLORMAP_AUTUMN', sensorData_COLORMAP_AUTUMN)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_AUTUMN.png', sensorData_COLORMAP_AUTUMN)
  #
  # sensorData_COLORMAP_BONE = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_BONE)
  # cv2.imshow('COLORMAP_BONE', sensorData_COLORMAP_BONE)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_BONE.png', sensorData_COLORMAP_BONE)
  #
  # sensorData_COLORMAP_CIVIDIS = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_CIVIDIS)
  # cv2.imshow('COLORMAP_CIVIDIS', sensorData_COLORMAP_CIVIDIS)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_CIVIDIS.png', sensorData_COLORMAP_CIVIDIS)
  #
  # sensorData_COLORMAP_COOL = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_COOL)
  # cv2.imshow('COLORMAP_COOL', sensorData_COLORMAP_COOL)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_COOL.png', sensorData_COLORMAP_COOL)
  #
  # sensorData_COLORMAP_HOT = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_HOT)
  # cv2.imshow('COLORMAP_HOT', sensorData_COLORMAP_HOT)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_HOT.png', sensorData_COLORMAP_HOT)
  #
  # sensorData_COLORMAP_HSV = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_HSV)
  # cv2.imshow('COLORMAP_HSV', sensorData_COLORMAP_HSV)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_HSV.png', sensorData_COLORMAP_HSV)
  #
  # sensorData_COLORMAP_INFERNO = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_INFERNO)
  # cv2.imshow('COLORMAP_INFERNO', sensorData_COLORMAP_INFERNO)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_INFERNO.png', sensorData_COLORMAP_INFERNO)
  #
  # sensorData_COLORMAP_JET = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_JET)
  # cv2.imshow('COLORMAP_JET', sensorData_COLORMAP_JET)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_JET.png', sensorData_COLORMAP_JET)
  #
  # sensorData_COLORMAP_MAGMA = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_MAGMA)
  # cv2.imshow('COLORMAP_MAGMA', sensorData_COLORMAP_MAGMA)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_MAGMA.png', sensorData_COLORMAP_MAGMA)
  #
  # sensorData_COLORMAP_OCEAN = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_OCEAN)
  # cv2.imshow('COLORMAP_OCEAN', sensorData_COLORMAP_OCEAN)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_OCEAN.png', sensorData_COLORMAP_OCEAN)
  #
  # sensorData_COLORMAP_PARULA = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_PARULA)
  # cv2.imshow('COLORMAP_PARULA', sensorData_COLORMAP_PARULA)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_PARULA.png', sensorData_COLORMAP_PARULA)
  #
  # sensorData_COLORMAP_PINK = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_PINK)
  # cv2.imshow('COLORMAP_PINK', sensorData_COLORMAP_PINK)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_PINK.png', sensorData_COLORMAP_PINK)
  #
  # sensorData_COLORMAP_PLASMA = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_PLASMA)
  # cv2.imshow('COLORMAP_PLASMA', sensorData_COLORMAP_PLASMA)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_PLASMA.png', sensorData_COLORMAP_PLASMA)
  #
  # sensorData_COLORMAP_RAINBOW = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_RAINBOW)
  # cv2.imshow('COLORMAP_RAINBOW', sensorData_COLORMAP_RAINBOW)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_RAINBOW.png', sensorData_COLORMAP_RAINBOW)
  #
  # sensorData_COLORMAP_SPRING = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_SPRING)
  # cv2.imshow('COLORMAP_SPRING', sensorData_COLORMAP_SPRING)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_SPRING.png', sensorData_COLORMAP_SPRING)
  #
  # sensorData_COLORMAP_SUMMER = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_SUMMER)
  # cv2.imshow('COLORMAP_SUMMER', sensorData_COLORMAP_SUMMER)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_SUMMER.png', sensorData_COLORMAP_SUMMER)
  #
  # sensorData_COLORMAP_TURBO = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_TURBO)
  # cv2.imshow('COLORMAP_TURBO', sensorData_COLORMAP_TURBO)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_TURBO.png', sensorData_COLORMAP_TURBO)
  #
  # sensorData_COLORMAP_TWILIGHT = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_TWILIGHT)
  # cv2.imshow('COLORMAP_TWILIGHT', sensorData_COLORMAP_TWILIGHT)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_TWILIGHT.png', sensorData_COLORMAP_TWILIGHT)
  #
  # sensorData_COLORMAP_TWILIGHT_SHIFTED = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_TWILIGHT_SHIFTED)
  # cv2.imshow('COLORMAP_TWILIGHT_SHIFTED', sensorData_COLORMAP_TWILIGHT_SHIFTED)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_TWILIGHT_SHIFTED.png', sensorData_COLORMAP_TWILIGHT_SHIFTED)
  #
  # sensorData_COLORMAP_VIRIDIS = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_VIRIDIS)
  # cv2.imshow('COLORMAP_VIRIDIS', sensorData_COLORMAP_VIRIDIS)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_VIRIDIS.png', sensorData_COLORMAP_VIRIDIS)
  #
  # sensorData_COLORMAP_WINTER = cv2.applyColorMap(sensorDataNormalized, cv2.COLORMAP_WINTER)
  # cv2.imshow('COLORMAP_WINTER', sensorData_COLORMAP_WINTER)
  # cv2.imwrite(outputPrefix + 'out/01-false-color/01-COLORMAP_WINTER.png', sensorData_COLORMAP_WINTER)


sensorDataNormalizedBlurred = cv2.bilateralFilter(sensorDataNormalized, 5, 21,
                                                  7)

(T, thresh) = cv2.threshold(sensorDataNormalizedBlurred, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# (T, thresh) = cv2.threshold(sensorDataNormalizedBlurred, 80, 255,
#                             cv2.THRESH_BINARY)
cv2.imshow("Threshold", thresh)
cv2.imwrite(
  outputPrefix + '00_-hist-normalized-blurred-threshold-simples.png',
  thresh)

contours, _ = cv2.findContours(thresh, mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_NONE)
# Determine center of gravity and orientation using Moments
M = cv2.moments(contours[0])
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# Display results
plt.figure()
plt.title("center")
plt.imshow(thresh, cmap='gray')
plt.scatter(center[0], center[1], marker="X")
plt.scatter(center[0], 0, marker="o")
plt.scatter(center[0], 480, marker="o")
plt.savefig(outputPrefix + "00-normalized-moments.png")
plt.show()

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
cv2.imshow("sensorDataNormalizedUnsharped", sensorDataNormalizedUnsharped)

# metade_template = sensorDataNormalized[center[1]:480, 0:640]
metade_template = sensorDataNormalizedUnsharped[0:480, center[0]:640]
cv2.imshow("metade template", metade_template)

# metade_template_flipped = cv2.flip(metade_template, 1)
# cv2.imshow("metade template flipped", metade_template_flipped)

# 2 recortar a metade da mascara e flip horizontal

metade_mask = thresh[0:480, center[0]:640]
cv2.imshow("metade mask", metade_mask)

metade_template_masked = cv2.bitwise_and(metade_template, metade_mask)
cv2.imshow("metade_template_masked", metade_template_masked)

primeira_metade_template = sensorDataNormalizedUnsharped[0:480, 0:center[0]]
primeira_metade_mask = thresh[0:480, 0:center[0]]
primeira_metade_template_masked = cv2.bitwise_and(primeira_metade_template, primeira_metade_mask)
cv2.imshow("primeira metade_template_masked", primeira_metade_template_masked)

cv2.imwrite(outputPrefix + "03-metade-template-esq.png", metade_template)
cv2.imwrite(outputPrefix + "03-metade-template-esq-masked.png", metade_template_masked)
cv2.imwrite(outputPrefix + "03-metade-template-dir.png", primeira_metade_template)
cv2.imwrite(outputPrefix + "03-metade-template-dir-masked.png", primeira_metade_template_masked)


# 3 usando a metade como template + mascara, busca por match na primeira metade
#   3.1 varias algoritmos de template match, que são bem inflexíveis
#   3.2 experimentar match a partir de features?


# cv2.namedWindow("template match original", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("template match result", cv2.WINDOW_AUTOSIZE)


# def MatchingMethod(param):
#   match_method = param
#
#   method_accepts_mask = (
#       cv2.TM_SQDIFF == match_method or match_method == cv2.TM_CCORR_NORMED)
#   if (method_accepts_mask):
#     result = cv2.matchTemplate(sensorDataNormalized, metade_template_flipped,
#                                match_method, None, metade_mask_inv_flipped)
#   else:
#     result = cv2.matchTemplate(sensorDataNormalized, metade_template_flipped,
#                                match_method)
#
#   cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
#   _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
#
#   if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
#     matchLoc = minLoc
#   else:
#     matchLoc = maxLoc
#
#   img_display = sensorDataNormalized.copy()
#   cv2.rectangle(img_display, matchLoc, (
#     matchLoc[0] + metade_template_flipped.shape[0],
#     matchLoc[1] + metade_template_flipped.shape[1]), 0, 2, 8, 0)
#   cv2.rectangle(result, matchLoc, (
#     matchLoc[0] + metade_template_flipped.shape[0],
#     matchLoc[1] + metade_template_flipped.shape[1]), 0, 2, 8, 0)
#   cv2.imshow("template match template", metade_template_flipped)
#   cv2.imshow("template match mask", metade_mask_inv_flipped)
#   cv2.imshow("template match original", img_display)
#   cv2.imshow("template match result", result)


# trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
# cv2.createTrackbar("Template Match Method", "template match original", 0, 5, MatchingMethod)


# 3b (feature matching, SHIFT)
#   pq o template match não deu certo

# def doSift():
#   cv = cv2
#   np = numpy
#
#   MIN_MATCH_COUNT = 4
#   img1 = metade_template_flipped.copy()
#   img2 = sensorDataNormalized.copy()
#
#   sift = cv.SIFT_create()
#
#   # find the keypoints and descriptors with SIFT
#   kp1, des1 = sift.detectAndCompute(img1, None)
#   kp2, des2 = sift.detectAndCompute(img2, None)
#
#   FLANN_INDEX_KDTREE = 1
#   index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#   search_params = dict(checks=50)
#   flann = cv.FlannBasedMatcher(index_params, search_params)
#   matches = flann.knnMatch(des1, des2, k=2)
#   # store all the good matches as per Lowe's ratio test.
#   good = []
#   for m, n in matches:
#     # if m.distance < 0.7 * n.distance:
#     if m.distance < 0.8 * n.distance:
#       good.append(m)
#
#   if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h, w = img1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
#       -1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#     img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
#   else:
#     print(
#       "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#     matchesMask = None
#
#   draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
#                      singlePointColor=None,
#                      matchesMask=matchesMask,  # draw only inliers
#                      flags=2)
#   img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
#   plt.imshow(img3, 'gray'), plt.show()


# doSift()

# 4 gerar imagem com a composição das duas para conferir visualmente o match
# 5 subtrair uma imagem da outra
#   5.1 posso acertar o range fazendo abs(subtract())
#     que é [0, -255] -> [0, 255]
#   5.2 posso fazer uma renormalização -255..255 -> 0.255 (me parece melhor)
# 6 color map nas imagens de 5.1 e 5.2
# bonus
# 7 analise de features na subtração
#   7.1 comparar features de healthy vs sick
# # bonus
# 8 tentar aplicar a mesma técnica nas outras imagens siméticas do protocolo
#  8.1 lateral 45 graus
#  8.2 lateral 90 graus
# # bonus
# 9 analise de features de lado esquerdo e direito
#  9.1 comparação entre imagens e entre sets healthy vs sick
# # bonus
# 10 extensão para outras imagens simétricas do protocolo
#   10.1 lateral 45 graus
#   10.1 lateral 90 graus


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

haralick_labels = mahotas.features.texture.haralick_labels[:-1]
haralick_features_metade = mahotas.features.haralick(metade_template_masked, return_mean=True, ignore_zeros=True)
haralick_features_primeira_metade = mahotas.features.haralick(primeira_metade_template_masked, return_mean=True, ignore_zeros=True)

haralick_features_diferenca = numpy.abs(numpy.subtract(haralick_features_metade, haralick_features_primeira_metade))

numpy.savetxt(outputPrefix + "haralick_labels.csv", [haralick_labels], delimiter=",", fmt='%s')
numpy.savetxt(outputPrefix + "haralick_features_metade.csv", [haralick_features_metade], delimiter=",", fmt='%s')
numpy.savetxt(outputPrefix + "haralick_features_primeira_metade.csv", [haralick_features_primeira_metade], delimiter=",", fmt='%s')
numpy.savetxt(outputPrefix + "haralick_features_diferenca.csv", [haralick_features_diferenca], delimiter=",", fmt='%s')

# cv2.waitKey(0)
cv2.destroyAllWindows()
