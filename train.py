import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    model = Sequential([
        VGG16(weights='imagenet', include_top=False, input_shape=(227, 227, 3)),
        # https://phamdinhkhanh.github.io/2020/05/31/CNNHistory.html xem về vgg16, ở đây chỉ lấy một phần vgg16, kĩ thuật này gọi là tranfer learning
        # đầu ra thu được ma trận dạng 7x7x512
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        # NUM_CLASSES = 4, kích thước cửa sổ là 1x1 (kích thước ma trận không thay đổi), vì vậy vẫn là ma trận 7x7, nhưng khi qua layer này thì thành 7x7x4
        # đầu ra thu được ma trận dạng 7x7x4
        Activation('relu'),
        GlobalAveragePooling2D(),
        # lấy trung bình của 4 ma trận 7x7, tưởng tượng như 7x7 thành 1x1 do nó lấy trung bình tất cả các phần từ trong ma trận 7x7
        # như vậy từ 7x7x4 thành 1x1x4 ( hoặc kích thước từ 7x7x4 thành 4)
        Activation('softmax')
        # lựa chọn softmax thì đầu ra sẽ có kết quả 0 hoặc 1
        # ví dụ kết quả là [1 0 0 0] thì là rock
        # tương tự có các kết quả cho "paper", "scissors", "none"
    ])
    return model


# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])


data, labels = zip(*dataset)
labels = list(map(mapper, labels))


'''
labels  one hot encoded
rock   [1,0,0,0]
paper   [0,1,0,0]
scissors    [0,0,1,0]
none    [0,0,0,1]
'''

# one hot encode the labels
labels = np_utils.to_categorical(labels) 
# Chuyển về dạng onehot để sự dụng loss='categorical_crossentropy'

# define the model
model = get_model()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(np.array(data), np.array(labels), epochs=10)

# save the model
model.save("rock-paper-scissors-model.h5")
