import numpy as np
from sklearn.preprocessing import Normalizer
import torch
from torchvision import models
from torchvision.utils import save_image
import cv2

import torch.nn.functional as F
from pathlib import Path
import argparse
import os
import sys
import matplotlib.pyplot as plt 
from config import CONFIG, Dataset,TargetModel
from datamodule import DataModule
import pandas as pd

directorio = os.path.join(Path.home(), "process_images")
sys.path.append(directorio) #para poder acceder a los ficheros como librerías (recuerda añadir __init__.py)
os.chdir(directorio) #para fijar este directorio como el directorio de trabajo

#NÚMERO DE IMÁGENES Y BBDD IMÁGENES
num_images = 5
data_images = 'mnist784_ref'

if __name__ == '__main__':
    g_easy = cv2.imread('Images/acum_g_easy.jpg')
    g_diff = cv2.imread('Images/acum_g.jpg')
    ig_easy = cv2.imread('Images/acum_ig_easy.jpg')
    ig_diff = cv2.imread('Images/acum_ig.jpg')

    g_total = (g_easy + g_diff) / 2
    ig_total = (ig_easy + ig_diff) / 2

    max_g = np.max(g_total)
    min_g = np.min(g_total)
    max_ig = np.max(ig_total)
    min_ig = np.min(ig_total)

    g_total_enhanced = (g_total - min_g) * (255.0/(max_g-min_g))
    ig_total_enhanced = (ig_total - min_ig) * (255.0/(max_ig-min_ig))
    image_shape = ig_total_enhanced.shape

    lista = [name for name, member in Dataset.__members__.items()]

    if data_images in lista:
        config = CONFIG
        dataset_enum = Dataset[data_images]
        data_module = DataModule(os.path.join(config.path_data, dataset_enum.value), batch_size = config.batch_size, 
            num_workers = config.NUM_WORKERS, 
            pin_memory=True, 
            dataset = dataset_enum)
        data_module.setup()

        dataset = data_module.fulldataset

       #load csv with difficulties
        if(os.path.exists("results_to_Carlos.csv")):
            datos = pd.read_csv("results_to_Carlos.csv")
        else:
            raise Exception("results_to_Carlos.csv does not exists")
        
        #Selección de imágenes fáciles
        datos_faciles = datos.sort_values(by = ['results'])
        datos_faciles = datos_faciles.loc[datos_faciles["results"] <= -4]
        if(np.shape(datos_faciles)[0] > num_images):
            datos_faciles = datos_faciles.sample(n=num_images)
        else:
            num_images = np.shape(datos_faciles)[0]
        
        datos_dificiles = datos.sort_values(by = ['results'], ascending = False)
        datos_dificiles = datos_dificiles.loc[datos_dificiles["results"] >= 4]
        if(np.shape(datos_dificiles)[0] > num_images):
            datos_dificiles = datos_dificiles.sample(n=num_images)
        else:
            num_images = np.shape(datos_dificiles)[0]
        #print(datos_faciles.head(5))
        #print(datos_dificiles.head(5))
        #sys.exit()

       #Muestra las etquetas seleccionadas y el valor límite de las etiquetas.
        contador = np.zeros(10)
        for etiqueta in datos_faciles.loc[:,"labels"]:
            contador[etiqueta] = contador[etiqueta] + 1

        print(contador)
        np.savetxt('easy_data.csv', contador, fmt="%d", delimiter=',')
        print(np.sum(contador))
        
        j=0

        for imagen in datos_faciles.loc[:,"id_image"] :
            j = j + 1
            print('the image number: {}'.format(j))
            img = dataset[imagen][0]
            # print(len(dataset))
            # print(img.shape)
            # print(img)
            # print(args.cuda)

            imgnp = img.numpy()
            imgnp = np.transpose(imgnp, (1,2,0))
            if(imgnp.shape[2] == 1):
                imgnp = np.squeeze(imgnp)
                imgnp = cv2.merge((imgnp, imgnp, imgnp))
            imgnp = cv2.normalize(imgnp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            imgnp_scaled = cv2.resize(np.uint8(imgnp), image_shape[0:2])
            image_name = "easy_{images}_{num}_.jpg".format(images = data_images, num = j)
            cv2.imwrite(os.path.join("Images", image_name), imgnp_scaled)

            imgnp_mapped = ig_total_enhanced * (imgnp_scaled/255)
            image_name = "easy_mapped_{images}_{num}_.jpg".format(images = data_images, num = j)
            cv2.imwrite(os.path.join("Images", image_name), imgnp_mapped)
        
        #Muestra las etquetas seleccionadas y el valor límite de las etiquetas.
        contador = np.zeros(10)
        for etiqueta in datos_dificiles.loc[:,"labels"]:
            contador[etiqueta] = contador[etiqueta] + 1

        print(contador)
        np.savetxt('diff_data.csv', contador, fmt="%d", delimiter=',')
        print(np.sum(contador))
    
        j=0

        for imagen in datos_dificiles.loc[:,"id_image"] :
            j = j + 1
            print('the image number: {}'.format(j))
            img = dataset[imagen][0]
            # print(len(dataset))
            # print(img.shape)
            # print(img)
            # print(args.cuda)

            imgnp = img.numpy()
            imgnp = np.transpose(imgnp, (1,2,0))
            if(imgnp.shape[2] == 1):
                imgnp = np.squeeze(imgnp)
                imgnp = cv2.merge((imgnp, imgnp, imgnp))
            imgnp = cv2.normalize(imgnp, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            imgnp_scaled = cv2.resize(np.uint8(imgnp), image_shape[0:2])
            image_name = "diff_{images}_{num}_.jpg".format(images = data_images, num = j)
            cv2.imwrite(os.path.join("Images", image_name), imgnp_scaled)

            imgnp_mapped = ig_total_enhanced * (imgnp_scaled/255)
            image_name = "diff_mapped_{images}_{num}_.jpg".format(images = data_images, num = j)
            cv2.imwrite(os.path.join("Images", image_name), imgnp_mapped)


    cv2.imwrite("Images/g_total_enhanced.jpg", g_total_enhanced)
    cv2.imwrite("Images/ig_total_enhanced.jpg", ig_total_enhanced)
