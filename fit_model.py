from keras.backend import flatten
import numpy as np
import os

import keras as k
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
data = []
labels = []






# def viewImage(image, name_of_window):
#     cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
#     cv2.imshow(name_of_window, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# img = cv2.imread('./img/1_0.jpg')
# img = cv2.resize(img, (256, 256))
# grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, grayImg = cv2.threshold(grayImg,210,255,cv2.ADAPTIVE_THRESH_MEAN_C)

#Обработка изображений
# path = os.listdir(str(os.getcwd())+'\img')
# print(path)

# for el in path:
#     img = cv2.imread('./img/'+el)
#     img = cv2.resize(img, (256, 256))
#     grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, grayImg = cv2.threshold(grayImg,210,255,cv2.ADAPTIVE_THRESH_MEAN_C)
#     labels.append(el.split('.')[-2].split('_')[-1]) 
#     data.append(grayImg)
# print(labels)


# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)

# x = np.array([0.3, 0.7, 0.9])
# y = np.array([0.5, 0.9, 1.0])


#Сохранение массива
# np.save('dataset', data)
# np.save('labels', labels)
# print("Массивы сохранены")

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
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
fit_result = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCH, batch_size=20) #EarlyStopping(monitor='val_acc', patience=3)

# testdata = []
# test_img = cv2.imread('./img/5_30.jpg')
# test_img = cv2.resize(test_img, (256, 256))
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# ret, test_img = cv2.threshold(test_img,210,255,cv2.ADAPTIVE_THRESH_MEAN_C)
# testdata.append(test_img)
# testdata = np.array(testdata, dtype='float') / 255.0
# predictions = model.predict(testdata)
# i = predictions.argmax(axis=1)[0]
# out = lb.classes_[i]
# print(out)
# predictions = model.predict(testX, batch_size=2)
# # print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))


# j = 0
# for el in predictions:
#     i = el.argmax()
#     out = lb.classes_[i]
#     print('Истинное значение: ', lb.classes_[testY.argmax(axis=1)[j]] ,' Предсказанное: ', out)
#     j+=1




# i = predictions.argmax(axis=1)[0]
# out = lb.classes_[i]
# print(out)
# print(predictions, testY[0])



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