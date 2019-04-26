# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:36:05 2019
Not:Egitim verisini tek dosya halinde tutma
@author: Yasir
"""

import os
import imageio
import numpy as np
from os import listdir
from keras.utils import to_categorical
from scipy.misc import imread, imresize



genislik, yukseklik = 224, 224
kanal_sayisi = 3 # 1: Grayscale, 3: RGB
etiket_sayisi = 6 # sınıf sayısı
drivePath = "drive/proje"
resimler_klasoru = os.path.join(drivePath,'dataset-resized')
veriseti_klasoru = os.path.join(resimler_klasoru)

def resmi_al(resimler_klasoru):
    resim = imageio.imread(resimler_klasoru)
    return resim

etiketler = listdir(resimler_klasoru) 
X, Y = [], []

for i, etiket in enumerate(etiketler):
    etiket_klasoru = os.path.join(resimler_klasoru, etiket)
    
    for resim_adi in listdir(etiket_klasoru):
        resim = resmi_al(os.path.join(etiket_klasoru, resim_adi))
        X.append(resim)
        Y.append(i)
    
    print(etiket)

X = np.array(X).astype('float32')/255.
X = X.reshape(X.shape[0], genislik, yukseklik, kanal_sayisi)
Y = np.array(Y).astype('float32')
Y = to_categorical(Y, etiket_sayisi)

  
if not os.path.exists(veriseti_klasoru):
    os.makedirs(veriseti_klasoru+'/')
 

np.save(os.path.join(veriseti_klasoru, 'resim.npy'), X)
np.save(os.path.join(veriseti_klasoru, 'category.npy'), Y)
