"""
Тестирование сети
"""
import os
import numpy as np
import keras as k
from sklearn.preprocessing import LabelBinarizer
import cv2

#Функция отображения изображения
def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


data = [] #Предварительно обработанные изображения
labels = []
imgs = [] #Изображения без обработки


"""
TODO перенести формирование массива изображений без обработки в prepare_img
"""
# #Загрузка изображений
# path = os.listdir(str(os.getcwd())+'/img')
# for el in path:
#     tempimg = cv2.imread(str(os.getcwd())+'/img/'+str(el))
#     tempimg = cv2.resize(tempimg, (800,800))
#     imgs.append(tempimg)
# print("Изображения загружены")   

# np.save('imgdata', imgs) #Сохранение массива изображений
imgs = np.load('videoimg.npy') #Загрузка массива изображений


data = np.load('videodata.npy') #Загрузка массива
labels = np.load('labels.npy') #Загрузка массива меток
model = k.models.load_model('./firstmodel') #Загрузка модели

lb = LabelBinarizer() #Кодирование меток строк с помощью чисел
labels = lb.fit_transform(labels) #Нахождение всех уникальных меток 


predictions = model.predict(data, batch_size=32, use_multiprocessing=True)
print("Распознавание завершено")

j = 0
for el in predictions:
    i = el.argmax()
    out = lb.classes_[i]
    text = "Prediction: {}".format(out)
    outimg = cv2.putText(imgs[j], text, (0,230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)
    # viewImage(imgs[j],'Images')
    cv2.imwrite(str(os.getcwd())+'/results/video/'+str(j)+'.jpg', imgs[j])
    # cv2.waitKey(0)
    j+=1
