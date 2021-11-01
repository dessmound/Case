#Обработка изображений

import os
import cv2
import numpy as np


data = []
labels = []

path = os.listdir(str(os.getcwd())+'\img')
print(path)

for el in path:
    img = cv2.imread('./img/'+el)
    img = cv2.resize(img, (256, 256)) #Изменение размера изображения 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Преобразование в оттенки серого
    ret, grayImg = cv2.threshold(grayImg,210,255,cv2.ADAPTIVE_THRESH_MEAN_C) #Увеличение яркости пикселя по порогу
    labels.append(el.split('.')[-2].split('_')[-1]) #Добавление показания индикатора в массив
    data.append(grayImg) #Добавление в массив значения яркости пикселей изображения (от 0 до 255)



data = np.array(data, dtype="float") / 255.0 #Преобразование интенсивности пиксели из диапазона от 0 до 255 в вещественный диапазон от 0 до 1
labels = np.array(labels)


# Сохранение массива
np.save('Имя массива', data)
np.save('Имя массива', labels)

print("Массивы сохранены")