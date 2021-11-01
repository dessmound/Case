
"""
Обучение сети

"""
import numpy as np
import keras as k
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

data = []
labels = []


#Загрузка массивов

data = np.load('dataset.npy') 
labels = np.load('labels.npy')

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.1, random_state=42) #Разбиение набора на тренировочный и проверочный

lb = LabelBinarizer() #Кодирование меток строк с помощью чисел
trainY = lb.fit_transform(trainY) #Нахождение всех уникальных меток 
testY = lb.transform(testY)


EPOCH = 25 #Количество эпох

# model = k.models.load_model('./firstmodel')#Загрузка модели
model = k.Sequential()
model.add(k.layers.Reshape((256,256,1), input_shape=data.shape[1:]))#Слой который переформировывает входные данные в заданную форму
model.add(k.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(data.shape[0],256,256,1)))
model.add(k.layers.MaxPooling2D((2, 2)))
model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(k.layers.MaxPooling2D((2, 2)))
model.add(k.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(k.layers.MaxPooling2D((2, 2)))
model.add(k.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(k.layers.MaxPooling2D((2, 2)))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(256, activation='relu'))
model.add(k.layers.Dense(64, activation='relu'))
model.add(k.layers.Dense(lb.classes_.size, activation='softmax'))
model.summary() #Отображение структуры сети
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) #Оптимизатор Adam; Функция потери категориальная кроссэнтропия
fit_result = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCH, batch_size=20) #Для остановки обучения при возрастании ошибки: EarlyStopping(monitor='val_acc', patience=3)



#Построение графика
N = np.arange(0, EPOCH)#эпохи
plt.style.use("ggplot")
plt.figure()
plt.plot(N, fit_result.history["loss"], label="потеря(тренировочный набор)")
plt.plot(N, fit_result.history["val_loss"], label="потеря(набор валидации)")
plt.plot(N, fit_result.history["accuracy"], label="точность(тренировочный набор)")
plt.plot(N, fit_result.history["val_accuracy"], label="точность(набор валидации)")
plt.title("Функция потери и точности (Нейронная сеть)")
plt.xlabel("Эпоха #")
plt.ylabel("Потеря/Точность")
plt.legend()
plt.savefig('test') #TODO изменить имя файла

model.save('firstmodel')#Сохранение модели