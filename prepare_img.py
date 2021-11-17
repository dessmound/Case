#Обработка изображений

import os
import cv2
import numpy as np
from numpy.core.defchararray import array
from numpy.lib.function_base import copy


data = []
labels = []
imgs = []

varPath = '/img3/fortest/1'

# path = os.listdir(str(os.getcwd())+'/img3/test1119')
path = os.listdir(str(os.getcwd())+varPath)
print(path)

shuffle_path = np.array(path)

# np.random.shuffle(shuffle_path)
# np.random.shuffle(shuffle_path)
# np.random.shuffle(shuffle_path)
print(shuffle_path)


for el in shuffle_path:
    img = cv2.imread('./img3/fortest/1/'+el)
    img = cv2.resize(img, (256, 256)) #Изменение размера изображения 
    imgs.append(img)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Преобразование в оттенки серого
    ret, grayImg = cv2.threshold(grayImg,210,255,cv2.ADAPTIVE_THRESH_MEAN_C) #Увеличение яркости пикселя по порогу
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', grayImg)
    # cv2.waitKey(0)
    labels.append(el.split('.')[-2].split('_')[-1]) #Добавление показания индикатора в массив
    data.append(grayImg) #Добавление в массив значения яркости пикселей изображения (от 0 до 255)



data = np.array(data, dtype="float") / 255.0 #Преобразование интенсивности пиксели из диапазона от 0 до 255 в вещественный диапазон от 0 до 1
labels = np.array(labels)


# Сохранение массива
# np.save('datatest1119', data)
# np.save('labelstest1119', labels)
# np.save('imgtest1119',imgs)


np.save('./img3/fortest/datatest1119', data)
np.save('./img3/fortest/labelstest1119', labels)
np.save('./img3/fortest/imgtest1119',imgs)
print("Массивы сохранены")