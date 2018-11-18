import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import Nest_Net
from losses import  dice_coef_loss_bce, dice_coef, hard_dice_coef, binary_crossentropy
from my_tools import rle_encoding, rle_decode
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import Sequence

# https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view
# path = 'PicsArt/data/'
path = 'data/dataset1'
path = '/media/danil/Data/Datasets/PicsArt/dataset1'
BATCH = 12
target_shape = (320,240)

def load_train_data(path):
    print('===LOAD DATA===')
    train_images = os.listdir(os.path.join(path, 'images_prepped_train'))
    train_images_list = [resize(imread(os.path.join(path, 'images_prepped_train', img)), target_shape) for img in train_images]
    train_mask_list = [resize(imread(os.path.join(path, 'annotations_prepped_train', img)), target_shape) for img in train_images]
    return np.array(train_images_list), np.expand_dims(np.array(train_mask_list),-1)

def load_test_data(path):
    print('===LOAD TEST DATA===')
    image_names = os.listdir(os.path.join(path, 'images_prepped_test'))
    test_images_list = [resize(imread(os.path.join(path, 'images_prepped_test', img)),target_shape) for img in image_names]
    test_mask_list = [resize(imread(os.path.join(path, 'annotations_prepped_test', img)), target_shape) for img in image_names]
    return np.array(test_images_list), np.expand_dims(np.array(test_mask_list),-1)

def make_predict(model):
    image_names, test_images_array = load_test_data(path)
    test_images_array = test_images_array / 255.
    print('===PREDICT===')
    predict_mask = model.predict(test_images_array, batch_size=1, verbose=1)
    return test_images_array, predict_mask, image_names

def create_train_image_generator(X_train, y_train):
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rotation_range=10,
                         width_shift_range=0.15,
                         height_shift_range=0.15,
                         zoom_range=0.2,
                         horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    image_generator = image_datagen.flow(X_train , seed=seed, batch_size=BATCH)
    mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=BATCH)
    train_generator = zip(image_generator, mask_generator)
    return train_generator

def create_callbaks(model_name='unet++.h5'):
    checkpoint = ModelCheckpoint('weights/' + model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    return [checkpoint]

if __name__ == '__main__':
    train_images, train_mask = load_train_data(path)
    train_mask = (train_mask == 8./255).astype(float)

    test_images, test_mask = load_test_data(path)
    test_mask = (test_mask == 8./255).astype(float)

    train_generator = create_train_image_generator(train_images, train_mask)

    model = Nest_Net(320, 240, 3)
    #model = load_model('weights/unet_with_car_data.h5', compile=False)
    model.compile(optimizer=Adam(1e-3, decay=1e-5), loss=dice_coef_loss_bce, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])
    callbacks = create_callbaks(model_name='unet_with_car_data.h5')

    print('===FIT MODEL===')
    model.fit_generator(train_generator,
                        steps_per_epoch = train_images.shape[0]/BATCH,
                        epochs=20,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=(test_images, test_mask))


    # x,y = next(train_generator)
    # plt.figure()
    # imshow(x[0])
    # plt.show()
    # plt.figure()
    # imshow(y[0,...,0])
    # plt.show()
    #
    # plt.figure()
    # imshow(train_images[100])
    # plt.show()
    # plt.figure()
    # imshow(train_mask[100,...,0])
    # plt.show()

