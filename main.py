import numpy as np
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import Nest_Net
from losses import  dice_coef_loss_bce, dice_coef, hard_dice_coef, binary_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import Sequence

path = 'PicsArt/data/'
BATCH = 4

def load_train_data(path):
    print('===LOAD DATA===')
    train_images = os.listdir(os.path.join(path, 'train'))
    train_images_list = [imread(os.path.join(path, 'train', img)) for img in train_images]
    train_mask_list = [imread(os.path.join(path, 'train_mask', img.split('.')[0]+'.png')) for img in train_images]
    return np.array(train_images_list), np.expand_dims(np.array(train_mask_list),-1)

def create_train_image_generator(X_train, y_train):
    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         rescale=1. / 255,
                         rotation_range=0,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1

    image_generator = image_datagen.flow(X_train , seed=seed, batch_size=BATCH)
    mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=BATCH)
    train_generator = zip(image_generator, mask_generator)
    return train_generator

def create_callbaks():
    checkpoint = ModelCheckpoint('PicsArt/weights/unet++.h5', monitor='val_dice_coef', mode='max', save_best_only=True)
    return [checkpoint]

if __name__ == '__main__':
    train_images, train_mask = load_train_data(path)
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_mask, test_size = 0.15, random_state = 17)
    X_val = X_val / 255.
    y_val = y_val / 255.

    train_generator = create_train_image_generator(X_train, y_train)
    model = Nest_Net(320, 240, 3)
    model.compile(optimizer=Adam(1e-4, decay=1e-5), loss=dice_coef_loss_bce, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])
    callbaacks = create_callbaks()

    model.fit_generator(train_generator,
                        steps_per_epoch = X_train.shape[0]/BATCH,
                        epochs=20,
                        callbacks=callbaacks,
                        validation_data=(X_val, y_val))

    # x,y = next(train_generator)
    # plt.figure()
    # imshow(x[0])
    # plt.show()
    # plt.figure()
    # imshow(y[0,...,0])
    # plt.show()

