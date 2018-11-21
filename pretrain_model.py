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
from main import create_train_image_generator

# https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view
# path = 'PicsArt/data/'
# path = 'data/dataset1'
path = '/media/danil/Data/Datasets/PicsArt/dataset1'
BATCH = 12
target_shape = (320, 240)
supervision = False

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

def create_callbaks(model_name='unet++.h5'):
    checkpoint = ModelCheckpoint('weights/' + model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    return [checkpoint]

if __name__ == '__main__':
    train_images, train_mask = load_train_data(path)
    train_mask = (train_mask == 8./255).astype(float)

    test_images, test_mask = load_test_data(path)
    test_mask = (test_mask == 8./255).astype(float)

    train_generator = create_train_image_generator((train_images*255).astype(np.uint8), (train_mask*255).astype(np.uint8), batch=BATCH, supervision = supervision)

    model = Nest_Net(320, 240, 3, deep_supervision=supervision)
    #model = load_model('weights/unet_with_car_data.h5', compile=False)

    if supervision:
        loss = {'output_1': binary_crossentropy,
                'output_2': binary_crossentropy,
                'output_3': binary_crossentropy,
                'output_4': dice_coef_loss_bce}

        val_data = (test_images, {'output_1': test_mask,
                                  'output_2': test_mask,
                                  'output_3': test_mask,
                                  'output_4': test_mask})
        metric = [{'output_4': dice_coef}, {'output_4': hard_dice_coef}, {'output_4': binary_crossentropy}]
        loss_weight = [0.25, 0.25, 0.5, 1.]
    else:
        loss = dice_coef_loss_bce
        val_data = (test_images, test_mask)
        metric = [dice_coef, hard_dice_coef, binary_crossentropy]
        loss_weight = [1.]

    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=metric, loss_weights=loss_weight)
    callbacks = create_callbaks(model_name='unet_with_car_data.h5')

    print('===FIT MODEL===')
    model.fit_generator(train_generator,
                        steps_per_epoch = train_images.shape[0]/BATCH,
                        epochs=20,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data)
    
    model = load_model('weights/unet_with_car_data_supervision.h5', compile=False)
    model.compile(optimizer=Adam(1e-4), loss=loss, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])
    model.fit_generator(train_generator,
                        steps_per_epoch = train_images.shape[0]/BATCH,
                        epochs=10,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data)

    x, y = next(train_generator)
    plt.figure()
    imshow(x[0])
    plt.show(block=False)
    plt.figure()
    imshow(y[0,...,0])
    plt.show(block=False)


