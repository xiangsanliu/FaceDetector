import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from load_data_set import loaddata, IMAGE_SIZE, resize_image


class Dataset:
    def __int__(self):

        # 训练数据
        self.train_images = None
        self.train_labels = None

        # 验证数据
        self.valid_images = None
        self.valid_labels = None

        # 测试数据
        self.test_images = None
        self.test_labels = None

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据并预处理
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=5):
        images, labels = loaddata("C:/Users/xiang/Pictures/face/")

        # 随机划分训练集、验证集
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        # 划分测试集
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 判断后端系统类型来调整数组维度
        if (K.image_dim_ordering() == 'th'):
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        # 输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid_samples')
        print(test_images.shape[0], 'test_samples')

        #模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量将类别标签进行one-hot编码使其向量化
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        #像素数据浮点化和归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels


class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=5):
        self.model = Sequential()

        #保留边界像素
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.input_shape, activation='relu'))#卷积层和激活函数
        ##输出(32, 64, 64)

        self.model.add(Convolution2D(32, 3, 3, activation='relu'))      #卷积层和激活函数
        #输出(32, 64, 64)

        self.model.add(MaxPooling2D(pool_size=(2, 2)))                                                       #池化层
        #输出(32, 32, 32)

        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
        #输出(64, 32, 32)

        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        #输出(64, 32, 32)

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #输出(64, 16, 16)

        self.model.add(Dropout(0.25))

        self.model.add(Flatten())   #数据从二维转为一维
        #输出64*16*16 =  16384

        #二层全连接神经网络 512*人的个数
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))

        self.model.add(Activation('softmax'))


        self.model.summary()

    def train(self, dataset, batch_size=20, nb_epoch=10, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # if not data_augmentation:
        self.model.fit(dataset.train_images, dataset.train_labels, batch_size=batch_size, nb_epoch=nb_epoch
                       , validation_data=(dataset.valid_images, dataset.valid_labels), shuffle=True)
        # else:
        #     datagen = ImageDataGenerator(
        #         featurewise_center=False,
        #         samplewise_center=False,
        #         featurewise_std_normalization=False,
        #         zca_whitening=False,
        #         rotation_range=20,
        #         width_shift_range=0.2,
        #         height_shift_range=0.2,
        #         horizontal_flip=True,
        #         vertical_flip=False
        #     )
        #
        #     datagen.fit(dataset.train_images)
        #
        #     self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels, batch_size=batch_size)
        #                              , samples_per_epoch=dataset.train_images.shape[0]
        #                              , nb_epoch=nb_epoch
        #                              , validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = 'C:/Users/xiang/Documents/face/me.face.model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    def face_predict(self, image):
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        image = image.astype('float32')
        image /= 255

        result = self.model.predict_proba(image)

        result = self.model.predict_classes(image)

        return result[0]


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load()

    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path="d:/face/me.face.model.h5")

    # model.load_model(file_path="C:/Users/xiang/Documents/face/me.face.model.h5")
    # model.evaluate(dataset)
