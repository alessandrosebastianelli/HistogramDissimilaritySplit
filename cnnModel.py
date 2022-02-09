from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import numpy as np
from time import time
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
import os
from tqdm import tqdm


class cnnModel:

    def __init__(self, name, train_paths, val_paths):
        self.name = name
        self.train_paths = train_paths
        self.val_paths = val_paths

        self.input_shape = (224,224,3)
        self.model = self.__build()

    def __build(self):
        kernel_size = (3,3)
        stride_size = (1,1)
        pool_size = (2,2)

        model = Sequential()
        model.add(Conv2D(32, kernel_size, strides=stride_size, padding='same', activation='relu', input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_img(self, path):
        # If the image is a tif file open it with rasterio 
        if '.tif' in path:
            with rasterio.open(path) as src:
                img = src.read()
                # By deafult rasterio puts the bands at the beginning, moving them at the end
                img = np.moveaxis(img, 0, -1) 
        # Otherwise open it with matplotlib
        else:
            img = np.array(Image.open(path))

        resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]), interpolation = cv2.INTER_AREA)
        
        return resized/255.0

    def __data_loader(self, paths, batch_size):
        b_in = np.zeros((batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        b_out = np.zeros((batch_size, 1))
        random.shuffle(paths)
        counter = 0

        while True:

            if counter >= len(paths) - batch_size:
                counter = 0
                random.shuffle(paths)

            for i in range(batch_size):

                if paths[counter].split('\\')[-2] == 'eruption':
                    b_out[i,0] = 1
                else:
                    b_out[i,0] = 0
                
                b_in[i,:,:,:] = self.load_img(paths[counter])

                counter = counter + 1
            
            yield b_in, b_out

    def train(self, epochs, batch_size):
        train_generator = self.__data_loader(self.train_paths, batch_size)
        val_generator = self.__data_loader(self.val_paths, batch_size)

        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

        init = time()
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(self.train_paths)//batch_size,
            validation_data = val_generator,
            validation_steps = len(self.val_paths)//batch_size,
            epochs = epochs,
            callbacks = [es]
        )
        elapsed_time = time() - init

        print('Training time: {} seconds'.format(elapsed_time))
        self.model.save(os.path.join('saved_models', self.name + '.h5'))

        return history

    def test(self):
        gt = np.zeros((len(self.val_paths)))
        pr = np.zeros((len(self.val_paths)))
        b_in = np.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        for i in tqdm(range(len(self.val_paths))):

            if self.val_paths[i].split('\\')[-2] == 'eruption':
                gt[i] = 1.0
            else:
                gt[i] = 0.0
                
            b_in[0,:,:,:] = self.load_img(self.val_paths[i])
            pr[i] = self.model.predict(b_in)

            predictions = (pr > 0.5).astype(int)

        return gt, pr, confusion_matrix(gt.astype(int), predictions), confusion_matrix(gt.astype(int), predictions, normalize='true')