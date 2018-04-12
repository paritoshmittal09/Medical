from __future__ import division, print_function
import numpy as np
import os
import sys

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import dicom
import shutil
import nrrd
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist
from sklearn.metrics.cluster import adjusted_rand_score

from keras import backend as K
K.set_image_data_format('channels_last')
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

#Describes the three network architectures
def standard_unet( img_rows, img_cols):

    inputs = Input((img_rows, img_cols,1))
		
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    print("conv1 shape:",conv1.shape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.8)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    return model

def reduced_unet( img_rows, img_cols, N = 2): # 2^N is the starting number of feature channels

    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(conv2)
    #drop2 = Dropout(0.8)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(conv4)
    #drop4 = Dropout(0.8)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(2**(N + 4), (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(2**(N + 4), (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(2**(N + 3),(2, 2), strides=(2, 2), padding='same')(conv5), conv4],axis=3)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(2**(N + 3), (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(2**(N + 2),(2, 2), strides=(2, 2), padding='same')(conv6), conv3],axis=3)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(2**(N + 1),(2, 2), strides=(2, 2), padding='same')(conv7), conv2],axis=3)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(2**N, (2, 2), strides=(2, 2),padding='same')(conv8), conv1],axis=3)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model

def simple_unet( img_rows, img_cols, N = 2): # 2^N is the starting number of feature channels

    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(2**N, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(conv2)
    #drop2 = Dropout(0.8)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(2**(N + 2), (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([Conv2D(2**(N+1), 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv3)),conv2],axis=3)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(up1)
    conv4 = Conv2D(2**(N + 1), (3, 3), activation='relu', padding='same')(conv4)


    up2 = concatenate([Conv2D(2**(N), 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv4)),conv1],axis=3)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(up2)
    conv5 = Conv2D(2**(N), (3, 3), activation='relu', padding='same')(conv5)

    conv6 = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[conv6])

    return model

#Function to the elastic deformation
def elastic_transform(image, alpha=0.0, sigma=0.25, random_state=None):
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

#fetches the dicom and nrrd data and stores them as numpy arrays
def load_create_data(img_rows=224, img_cols=224, load_path='../data/', save_path='../data/'):

    for direc in ['train', 'test', 'validate']: #change train, test, and validate to the data sub-folder names
        PathDicom = load_path + direc + '/'
        dcm_dict = dict()
        for dirName, subdirList, fileList in os.walk(PathDicom):
            if any(".dcm" in s for s in fileList):
                ptn_name = dirName.split('/')[3] #retrieve the patient name from the filename
                fileList = [x for x in fileList if '.dcm' in x]
                indice = [ int( fname[:-4] ) for fname in fileList]
                imgs = np.zeros( [indice[-1]+1, img_rows, img_cols])
                for filename in np.sort(fileList):
                    img = dicom.read_file(os.path.join(dirName,filename)).pixel_array.T
                    img = equalize_hist( img.astype(int) ) #histogram equalization
                    img = resize( img, (img_rows, img_cols), preserve_range=True)
                    imgs[int(filename[:-4])] = img
                dcm_dict[ptn_name] = imgs
        imgs = []
        img_masks = []
        for patient in dcm_dict.keys():
            for fnrrd in os.listdir(PathDicom):
                if fnrrd.startswith(patient) and fnrrd.endswith('nrrd'): #find the nrrd corresponding to the current dicom file
                    masks = np.rollaxis(nrrd.read(PathDicom + fnrrd)[0], 2)
                    rescaled = np.zeros( [ len(masks), img_rows, img_cols])
                    for mm in range(len(rescaled)):
                        rescaled[mm] = resize( masks[mm], (img_rows, img_cols), preserve_range=True)/2.0 #re-scale the mask data to 0 to 1 and resize to match the resized dicom image
                    masks = rescaled.copy()

                    #Check if the dimension of the masks and the images match. Use the data only if they do.
                    if len(dcm_dict[patient]) != len(masks) :
                        print('Dimension mismatch for {:s} in folder {:s}'.format(patient, direc))
                    else:
                        img_masks.append(masks)
                        imgs.append( dcm_dict[patient] )
                    break

        imgs = np.concatenate(imgs, axis=0).reshape(-1, img_rows, img_cols, 1)
        img_masks = np.concatenate(img_masks, axis=0).reshape(-1, img_rows, img_cols, 1)

        #Binary classification for now, using 0.45 (0.9 in the original mask) as the threshold
        img_masks = np.array(img_masks>0.45, dtype=int)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        #processing done, now save the data
        np.save(save_path + direc + "_" + str(img_rows) + '.npy', imgs)
        np.save(save_path + direc + "_" + str(img_rows) + '_masks.npy', img_masks)

#defines dice coefficient based custom loss function for use with training
def dice_coef(y_true, y_pred, smooth=1.0): #compute the dice coefficient
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):
    intersection = y_true*y_pred
    return ( 2. * intersection.sum(axis=axis) +smooth)/ (y_true.sum(axis=axis) + y_pred.sum(axis=axis) +smooth )

def load_data():
    #change paths to reflect the locations where the data has been saved using load_create_data()
    X_train = np.load('../data/train_224.npy')
    y_train = np.load('../data/train_224_masks.npy')
    X_val = np.load('../data/validate_224.npy')
    y_val = np.load('../data/validate_224_masks.npy')

    return X_train, y_train, X_val, y_val

def step_decay(epoch): #learning rate decay for the Adam solver
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * drop**int((1 + epoch) / epochs_drop)
    return lrate

def keras_fit_generator(n_imgs=10**4, batch_size=64):
    
    X_train, y_train, X_val, y_val = load_data()
    img_rows = X_train.shape[1]
    img_cols = img_rows

    #creating generator arguments for data augmentation
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        preprocessing_function=elastic_transform)

    #we create two instances with the same arguments
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1 #to maintain the mapping of masks to images after shuffling

    image_datagen.fit(X_train,seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    #perform data augmentation
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    #instantiate and configure network
    model = standard_unet(img_rows, img_cols) #To use simple_unet or reduced_unet, an additional parameter N can be passed to change the number of starting feature channels
    #model.load_weights('../data/weights_224_unet.h5') #To preload the weights in the nextwork for training
    model.summary()
    model_checkpoint = ModelCheckpoint('../data1/standard_224_cross.h5', monitor='loss',verbose=1, save_best_only=True)

    lrate = LearningRateScheduler(step_decay)

    model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy']) #using binary_crossentropy loss provides better results than dice_coefficient

    #begin model training
    model.fit_generator( train_generator,
                        steps_per_epoch=n_imgs//batch_size,
                        epochs=30,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=[model_checkpoint]
                         )

def calc_predictions(X_test,y_test,location): #computes the predicted mask for all test samples using the trained weights
    img_rows = X_test.shape[1]
    img_cols = img_rows
    model = standard_unet(img_rows, img_cols) #change standard_unet to reflect the network of choice
    model.load_weights(location)
    model.summary()
    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
    y_test = y_test.astype('float32')
    y_pred = model.predict( X_test, verbose=1)
    y_pred[y_pred<1e-1] = 0
    y_pred[y_pred>1e-1] = 1  
    return y_pred,y_test

def jaccard_index(y_test,y_pred,img_rows,img_cols): #computes the Jaccard index for all test samples
    jaccard_i=0
    for i in range(0,len(y_test)):
        A = y_pred[i].reshape(img_rows, img_cols)
        B = y_test[i].reshape(img_rows, img_cols)
        intersection = np.sum(np.multiply(A,B))
        union = np.sum(np.sum(A)) + np.sum(np.sum(B)) - intersection
        axis = tuple( range(1, y_test.ndim ) )
        indice = np.nonzero( y_test.sum(axis=axis) )[0]
        sample =len(indice)
        if union == 0:
            continue
        else:
            jaccard_i = jaccard_i + (intersection/union)
    return (jaccard_i/sample)

def dice_coefficient(y_test,y_pred,img_rows,img_cols): #computes the dice coefficient for all test samples
    J = jaccard_index(y_test,y_pred,img_rows,img_cols)
    return (2*J)/(1+J)

def pixel_error(y_test,y_pred,img_rows,img_cols): #computes the pixel error for all test samples
    pro = (img_rows * img_cols)
    pixel_accuracy = 0
    for i in range(0,len(y_test)):
        A = y_pred[i].reshape(img_rows, img_cols)
        B = y_test[i].reshape(img_rows, img_cols)
        A = A.flatten()
        B = B.flatten()
        pixel_accuracy = pixel_accuracy + (np.sum(np.logical_xor(A,B))/pro)

    return (pixel_accuracy/len(y_test))

def rand_error(y_test,y_pred,img_rows,img_cols): #computes the rand error for all test samples
    rand_accuracy = 0
    pro = (img_rows * img_cols)
    for i in range(0,len(y_test)):
        A = y_pred[i].reshape(img_rows*img_cols)
        B = y_test[i].reshape(img_rows*img_cols)
        rand_accuracy = rand_accuracy + adjusted_rand_score(A,B)
    return 1-(rand_accuracy/len(y_test))

def main(argv):
    if argv[1] == '-l':
        #load and create the preprocessed data
        print('Loading and preprocessing data...')
        load_create_data()
    elif argv[1] == '-t':
        #perform training
        print('Starting training...')
        keras_fit_generator(n_imgs=15*10**4,batch_size=16) #n_imgs is the number of training samples after data augmentation. Change batch size as per available memory
    elif argv[1] == '-e':
        #perform the evaluation
        print('Evaluating...')
        X_test = np.load('../data/test_224.npy') #change the location to the test data created using load_create_data()
        y_test = np.load('../data/test_224_masks.npy')
        img_rows = X_test.shape[1]
        img_cols = img_rows
        weights_location = "../data/standard_224_cross.h5" #change the path to the weights stored using training
        y_pred,y_test = calc_predictions(X_test,y_test,weights_location)
        print('Jaccard Index: {:.2f}'.format(jaccard_index(y_test,y_pred,img_rows,img_cols)))
        print('Dice Coefficient: {:.2f}'.format(dice_coefficient(y_test,y_pred,img_rows,img_cols)))
        print('Pixel error: {:.2f}'.format(pixel_error(y_test,y_pred,img_rows,img_cols)))
        print('Rand Error: {:.2f}'.format(rand_error(y_test,y_pred,img_rows,img_cols)))
    else:
        print('Parameter not recognized. Use -l to load and save the preprocessed data, -t to train, and -e to evaluate.')
        sys.exit()

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: python main.py <-l/-t/-e>')
        sys.exit()
    import time
    start = time.time()
    main(sys.argv)
    end = time.time()
    if end-start > 60:
        print('Elapsed time: {:.2f} minutes.'.format(round((end-start)/60, 2)))
    else:
        print('Elapsed time: {:.2f} seconds.'.format(round((end-start), 2)))
