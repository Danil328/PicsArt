import numpy as np
from skimage.io import imread, imshow
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import Nest_Net
from losses import  dice_coef_loss_bce, dice_coef, hard_dice_coef, binary_crossentropy
from my_tools import rle_encoding, rle_decode, rle_encode
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import Sequence

# path = 'PicsArt/data/'
path = '/media/danil/Data/Datasets/PicsArt/data/'
BATCH = 12

def load_train_data(path):
    print('===LOAD DATA===')
    train_images = os.listdir(os.path.join(path, 'train'))
    train_images_list = [imread(os.path.join(path, 'train', img)) for img in train_images]
    train_mask_list = [imread(os.path.join(path, 'train_mask', img.split('.')[0]+'.png')) for img in train_images]
    return np.array(train_images_list), np.expand_dims(np.array(train_mask_list),-1)

def load_test_data(path):
    print('===LOAD TEST DATA===')
    image_names = os.listdir(os.path.join(path, 'test'))
    test_images_list = [imread(os.path.join(path, 'test', img)) for img in image_names]
    return image_names, np.array(test_images_list)

def make_predict(model):
    image_names, test_images_array = load_test_data(path)
    test_images_array = test_images_array / 255.
    print('===PREDICT===')
    predict_mask = model.predict(test_images_array, batch_size=1, verbose=1)
    return test_images_array, predict_mask, image_names

def create_submission(image_names, predicted_mask, threshold=0.5):
    print('===CREATE SUBMISSION===')
    predicted_mask[predicted_mask >= threshold] = 1
    predicted_mask[predicted_mask < threshold] = 0
    rle_mask = [rle_encode(x) for x in predicted_mask]
    sub = pd.DataFrame()
    sub['image'] = image_names
    sub['image'] = sub['image'].map(lambda x: x.split('.')[0])
    sub['rle_mask'] = rle_mask
    sub.to_csv('submission/submission.csv', index=False)

def create_train_image_generator(X_train, y_train):
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rescale=1. / 255,
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
    checkpoint = ModelCheckpoint('weights/' + model_name, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1)
    return [checkpoint]

if __name__ == '__main__':
    train_images, train_mask = load_train_data(path)
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_mask, test_size = 0.15, random_state = 17)
    X_val = X_val / 255.
    y_val = y_val / 255.

    train_generator = create_train_image_generator(X_train, y_train)
    callbacks = create_callbaks()
    #model = Nest_Net(320, 240, 3)
    model = load_model('weights/unet_with_car_data2.h5', compile=False)
    #model = load_model('weights/unet++2.h5', compile=False)
    model.compile(optimizer=Adam(1e-3, decay=1e-5), loss=dice_coef_loss_bce, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])


    print('===FIT MODEL===')
    model.fit_generator(train_generator,
                        steps_per_epoch = X_train.shape[0]/BATCH,
                        epochs=20,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=(X_val, y_val),
                        initial_epoch=0)

    # x,y = next(train_generator)
    # plt.figure()
    # imshow(x[0])
    # plt.show()
    # plt.figure()
    # imshow(y[0,...,0])
    # plt.show()

    model = load_model('weights/unet++2.h5', compile = False)
    test_image_array, predicted_mask, test_image_names = make_predict(model)
    create_submission(test_image_names, predicted_mask, threshold=0.5)

    # plt.figure()
    # imshow(test_image_array[0])
    # plt.show()
    # plt.figure()
    # imshow(predicted_mask[0,...,0])
    # plt.show()