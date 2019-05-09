# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:41:28 2019

@author: Yasir
"""

import os
import numpy as np
import imageio
from os import listdir

prepath=os.path.join(os.getcwd(),'224_224_1')
savePath = os.path.join(os.getcwd(),'224_224_train_test')
trainPath= os.path.join(savePath,'train')
testPath = os.path.join(savePath,'test')

etiketler = listdir(prepath)

def resmi_al(resimler_klasoru):
    resim = imageio.imread(resimler_klasoru)
    return resim

def resmi_kaydet(resimler_klasoru, resim):
    imageio.imsave(resimler_klasoru, resim)


for i, etiket in enumerate(etiketler):
    etiketPath = os.path.join(prepath,etiket)
    resimler = listdir(etiketPath)
    np.random.shuffle(resimler)
    kesme = int(len(resimler)*0.8)
    train, test = resimler[:kesme], resimler[kesme:]
    
    for r, rsm in enumerate(train):
        resim= resmi_al(os.path.join(etiketPath,rsm))
        try :
            os.makedirs(os.path.join(trainPath, etiket))
        except OSError:
            if not os.path.isdir(os.path.join(trainPath, etiket)):
                raise
        imageio.imsave(os.path.join(os.path.join(trainPath, etiket), rsm), resim)
    
    
    for r, rsm in enumerate(test):
        resim= resmi_al(os.path.join(etiketPath,rsm))
        try :
            os.makedirs(os.path.join(testPath, etiket))
        except OSError:
            if not os.path.isdir(os.path.join(testPath, etiket)):
                raise
        imageio.imsave(os.path.join(os.path.join(testPath, etiket), rsm), resim)
        
        
        
        
        
        
