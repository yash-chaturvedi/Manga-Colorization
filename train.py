import os
import cv2
import numpy as np
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
import matplotlib.pyplot as plt
from keras.regularizers import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from time import time
from keras.preprocessing.image import ImageDataGenerator

fixed_seed_num = 1234
np.random.seed(fixed_seed_num)
tf.random.set_seed(fixed_seed_num)
x_shape = 512
y_shape = 512

def generator_model():
#     encoder
    generator_input = Input(batch_shape=(None,x_shape,y_shape, 1), name='generator_input')
    generator_input_normalized = BatchNormalization()(generator_input)
    
    conv1_32 = Conv2D(16,kernel_size=(3,3),strides=(1,1),padding='same',activation='elu',kernel_regularizer=l2(0.001))(generator_input)
    conv1_32 = BatchNormalization()(conv1_32)
    
    conv2_64 = Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv1_32)
    conv2_64 = Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv2_64)    
    conv2_64 = MaxPooling2D(pool_size=(2,2),padding="same")(conv2_64)
    conv2_64 = BatchNormalization()(conv2_64)
    
    conv3_128 = Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv2_64)
    conv3_128 = Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv3_128)
    conv3_128 = MaxPooling2D(pool_size=(2,2),padding="same")(conv3_128)
    conv3_128 = BatchNormalization()(conv3_128)
    
    conv4_256 = Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv3_128)
    conv4_256 = Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv4_256)
    conv4_256 = MaxPooling2D(pool_size=(2,2),padding="same")(conv4_256)
    conv4_256 = BatchNormalization()(conv4_256)
    
    conv5_512 = Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv4_256)
    conv5_512 = Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv5_512)
    conv5_512 = MaxPooling2D(pool_size=(2,2),padding="same")(conv5_512)
    conv5_512 = BatchNormalization()(conv5_512)
    
    conv6_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv5_512)
    conv6_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv5_512)
    conv6_512 = MaxPooling2D(pool_size=(2,2),padding="same")(conv6_512)
    conv6_512 = BatchNormalization()(conv6_512)
    
    conv7_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv6_512)
    conv7_512 = BatchNormalization()(conv7_512)
    
#     decoder
    conv8_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(conv7_512)
    conv8_512 = BatchNormalization(axis=1)(conv8_512)
    
    deconv9_512 = Conv2DTranspose(512,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=l2(0.001))(conv8_512)
    deconv9_512 = BatchNormalization()(deconv9_512)
    deconv9_512 = Concatenate()([deconv9_512,conv5_512])
    deconv9_512 = Conv2D(512,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv9_512)
    deconv9_512 = BatchNormalization()(deconv9_512)
    
    deconv10_256 = Conv2DTranspose(256,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=l2(0.001))(deconv9_512)
    deconv10_256 = BatchNormalization()(deconv10_256)
    deconv10_256 = Concatenate()([deconv10_256,conv4_256])
    deconv10_256 = Conv2D(256,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv10_256)
    deconv10_256 = BatchNormalization()(deconv10_256)
    
    deconv11_128 = Conv2DTranspose(128,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=l2(0.001))(deconv10_256)
    deconv11_128 = Concatenate()([deconv11_128,conv3_128])
    deconv11_128 = Conv2D(128,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv11_128)
    
    deconv12_64 = Conv2DTranspose(64,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=l2(0.001))(deconv11_128)
    deconv12_64 = Concatenate()([deconv12_64,conv2_64])
    deconv12_64 = Conv2D(64,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv12_64)
    
    deconv13_32 = Conv2DTranspose(32,kernel_size=(3,3),padding='same',activation='elu',strides=(2,2),kernel_regularizer=l2(0.001))(deconv12_64)
    deconv13_32 = Concatenate()([deconv13_32,conv1_32])
    deconv13_32 = Conv2D(32,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv13_32)
    
    deconv14_16 = Conv2DTranspose(16,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv13_32)
    deconv14_16 = Conv2D(16,kernel_size=(3,3),padding='same',activation='elu',kernel_regularizer=l2(0.001))(deconv14_16)
    
    output = Conv2D(3,kernel_size=(1,1),padding='same',activation='relu')(deconv14_16)
    
    model = Model(inputs=generator_input,outputs=output)
    
    return model

def discriminator_model():
    
    generator_input = Input(batch_shape=(None, x_shape, y_shape, 1), name='generator_output')
    generator_output = Input(batch_shape=(None, x_shape, y_shape, 3), name='generator_input')
    
    input1 = BatchNormalization()(generator_input)
    input2 = BatchNormalization()(generator_output)
    
    convi = Conv2D(32,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=l2(0.001))(generator_input)
    convi = BatchNormalization()(convi)
    
    convo = Conv2D(32,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=l2(0.001))(generator_output)
    convo = BatchNormalization()(convo)

    
    convi = Conv2D(64,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=l2(0.001))(convi)
    convi = BatchNormalization()(convi)
    
    convo = Conv2D(64,kernel_size=(3,3),activation='elu',padding='same',kernel_regularizer=l2(0.001))(convo)
    convo = BatchNormalization()(convo)

    
    convi = Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(convi)
    convo = Conv2D(64,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(convo)
    
    conv = Concatenate()([convi,convo])
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(128,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same',kernel_regularizer=l2(0.001))(conv)
    conv = BatchNormalization()(conv)
    
    conv = Conv2D(256,kernel_size=(3,3),strides=(2,2),activation='elu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    conv = Flatten()(conv)
    conv = Dropout(0.5)(conv)
    
    conv = Dense(100,activation='elu')(conv)
    conv = Dropout(0.5)(conv)
    
    output = Dense(1,activation='sigmoid')(conv)
    
    model = Model(inputs=([generator_input,generator_output]),outputs=[output])
    
    return model

def cGAN_model(generator,discriminator):
    
    discriminator.trainable = False
    model = Model(inputs=generator.inputs,outputs=[discriminator([generator.input,generator.output]), generator.output])
    
    return model

from keras import losses
def custom_loss(y_true,y_pred):
    
    cosine = losses.cosine_proximity(y_true,y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    l = (cosine)+mle
    
    return l

def custom_loss_2(y_true,y_pred):
    cosine = losses.cosine_proximity(y_true,y_pred)
    mse = losses.mean_squared_error(y_true, y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    l = (cosine+1)*mse+mle
    return l

def train(gen,disc,cGAN,gray,rgb,gray_val,rgb_val,batch):
    samples = len(rgb)
    gen_image = gen.predict(gray, batch_size=16)   
    gen_image_val = gen.predict(gray_val, batch_size=8)
    inputs = np.concatenate([gray, gray])
    outputs = np.concatenate([rgb, gen_image])
    y = np.concatenate([np.ones((samples, 1)), np.zeros((samples, 1))])
    disc.fit([inputs, outputs], y, epochs=1, batch_size=4)
    disc.trainable = False
    
    cGAN.fit(gray, [np.ones((samples, 1)), rgb], epochs=1, batch_size=batch,validation_data=[gray_val,[np.ones((val_samples,1)),rgb_val]])
    disc.trainable = True

gen = generator_model()
gen.summary()

disc = discriminator_model()
disc.summary()

cGAN = cGAN_model(gen, disc)
cGAN.summary()

disc.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08), metrics=['accuracy'])

cGAN.compile(loss=['binary_crossentropy',custom_loss_2], loss_weights=[5, 100], optimizer=Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
cGAN.load_weights("D:\Jupyter\cGAN_model.h5")

dataset = 'Manga_Colorization/datasets/train/' 
val_data = 'Manga_Colorization/datasets/validation/'
store = 'Result/'
store2 = 'Result/store/'
samples = len(os.listdir(dataset))
# samples = 70
val_samples = len(os.listdir(val_data))
# model_path = store+str(1200)+'.h5'
# samples = 6
rgb = np.zeros((samples, x_shape, y_shape, 3))
gray = np.zeros((samples, x_shape, y_shape, 1))
rgb_val = np.zeros((val_samples, x_shape, y_shape, 3))
gray_val = np.zeros((val_samples, x_shape, y_shape, 1))
y_train = np.zeros((samples,1))
# print(os.listdir(dataset)[0:5])
for i, image in enumerate(os.listdir(dataset)[:samples]):
    I = cv2.imread(dataset+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb[i] = I;
    gray[i] = J

for i, image in enumerate(os.listdir(val_data)[:val_samples]):
    I = cv2.imread(val_data+image)
    I = cv2.resize(I, (x_shape, y_shape))
    J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#     print(J.shape)
    J = J.reshape(J.shape[0], J.shape[1], 1)
    rgb_val[i] = I; gray_val[i] = J
    
datagen = ImageDataGenerator(zoom_range=0.2,fill_mode='wrap',horizontal_flip=True,vertical_flip=True,
                            rotation_range=15)
datagen.fit(rgb)

epochs = 1
b=1
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(rgb, y_train, batch_size=samples):
        for i in range(len(x_batch)):
            gray[i] = cv2.cvtColor(x_batch[i], cv2.COLOR_BGR2GRAY).reshape((x_shape,y_shape,1))
        #train(gen,disc,cGAN,gray,x_batch,gray_val,rgb_val,b)
        params = (gen, disc, cGAN, gray,  x_batch, gray_val, rgb_val, 1)
        train(*params)
        batches += 1
        if batches >= 1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    if e%5 == 0:
        cGAN.save_weights(store2+str(e)+'.h5') 
    gen_image_val = gen.predict(gray_val, batch_size=8)
    if e%1 == 0: 
        for j in range(val_samples):
            if not os.path.exists(store2+str(j)+'/'):
                os.mkdir(store2+str(j)+'/')
            cv2.imwrite(store2+str(j)+'/'+str(e)+'.jpg', gen_image_val[j])