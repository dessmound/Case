"""
Тестирование сети
"""

import numpy as np
import keras as k
from sklearn.metrics import LabelBinarizer




data = []
labels = []

data = np.load('dataset.npy') #Загрузка массива
labels = np.load('labels.npy') #Загрузка массива
model = k.models.load_model('./firstmodel') #Загрузка модели

lb = LabelBinarizer() #Кодирование меток строк с помощью чисел
labels = lb.fit_transform(labels) #Нахождение всех уникальных меток 


predictions = model.predict(data, batch_size=20)
j = 0
for el in predictions:
    i = el.argmax()
    out = lb.classes_[i]
    print('Истинное значение: ', lb.classes_[data.argmax(axis=1)[j]] ,' Предсказанное: ', out)
    j+=1
