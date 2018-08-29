from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import glob
import cv2

def cnn_autoencoder():
    input_img = Input(shape = (256,256,3))

    #encoder
    encoded1 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img) #(256, 256, 128)
    encoded2 = MaxPooling2D((2, 2), padding='same')(encoded1)
    encoded3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded2) # (128, 128, 64)
    encoded4 = MaxPooling2D((2, 2), padding='same')(encoded3)
    encoded5 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded4) # (64, 64, 32)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded5)
    
    #decoder
    decoded1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded) #(32, 32, 32)
    decoded2 = UpSampling2D((2, 2))(decoded1)
    decoded3 = Conv2D(64, (3, 3), activation='relu', padding='same')(decoded2)  #(64, 64, 64)
    decoded4 = UpSampling2D((2, 2))(decoded3)
    decoded5 = Conv2D(128, (3, 3), activation='relu',padding='same')(decoded4) # (128, 128, 128))
    decoded6 = UpSampling2D((2, 2))(decoded5)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded6) #(256, 256, 3))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

if __name__ == '__main__':
    x_train = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/train/*.png')])
    x_test = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/test/*.png')])
    x_val = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/val/*.png')])

    x_train = x_train.reshape(-1, 256, 256, 3)
    x_test = x_test.reshape(-1,256,256,3)
    x_val = x_val.reshape(-1,256,256,3)

    autoencoder = cnn_autoencoder()
    autoencoder.fit(x_train,x_train, epochs=3, validation_data=(x_test, x_test))
    # restored_imgs = autoencoder.predict(x_test)
    #
    # for i in range(5):
    # plt.imshow(x_test[i].reshape(256, 256))
    # plt.show()
    #
    # plt.imshow(restored_imgs[i].reshape(256, 256))
    # plt.show()
