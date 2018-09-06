import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib
matplotlib.use('Agg') #for AWS only
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from sklearn.cluster import KMeans

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cnn_autoencoder():
    '''Returns CNN autoencoder.'''

    input_img = Input(shape = (256,256,3))

    #encoder
    encoded1 = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img) #(256, 256, 128)
    pool1 = MaxPooling2D((2, 2), padding='same')(encoded1)
    encoded2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) # (128, 128, 64)
    pool2 = MaxPooling2D((2, 2), padding='same')(encoded2)
    encoded3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2) # (64,64,32)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded3) # (32,32,32)

    #decoder
    decoded1 = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  #(32,32,32)
    up1 = UpSampling2D((2, 2))(decoded1)
    decoded2 = Conv2D(64, (3, 3), activation='relu',padding='same')(up1) # (64,64,64)
    up2 = UpSampling2D((2, 2))(decoded2)
    decoded3 = Conv2D(128, (3, 3), activation='relu',padding='same')(up2) # (128, 128, 128)
    up3 = UpSampling2D((2, 2))(decoded3)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3) #(256, 256, 3))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

def get_compressed_images(model,X,compressed_layer):
    '''Returns reshaped array of compressed images in prep for clustering.

    Input:
    model = autoencoder model
    X = 4D array of images to compress
    compressed_layer = integer of compressed layer

    Output:
    2D array of compressed images
    '''
    get_compressed_output = K.function([model.layers[0].input], [model.layers[compressed_layer].output])
    X_compressed = get_compressed_output([X])[0]
    X_compressed = X_compressed.reshape(X_compressed.shape[0],X_compressed.shape[1]*X_compressed.shape[2]*X_compressed.shape[3])
    return X_compressed

def cluster_compressed(X_compressed):
    '''Fits KMeans model to the compressed images and returns kmeans model and image labels.'''

    kmeans = KMeans(n_clusters=50, n_jobs=-1)
    kmeans.fit(X_compressed)

    labels = kmeans.labels_
    return kmeans, labels

def get_kmeans_rec(item_index, kmeans, og_X, num_recs,filepath=None):
    '''Input:
    item_index = integer, index of item for which to get recs
    kmeans = kmeans model
    og_X = original image arrays
    num_recs = number of recommendations

    Output:
    recs = indices of recommendations if no filepath specified
    saved images if filepath
    '''

    labels = kmeans.labels_
    cluster_label = kmeans.labels_[item_index]
    cluster_members = og_X[labels == cluster_label]
    indices = np.random.choice(len(cluster_members), num_recs, replace=False)
    recs = cluster_members[indices]

    #show recs
    for rec, i in zip(recs,range(num_recs)):
        plt.imshow(rec.reshape(256,256,3))
        if filepath:
            plt.savefig('{}rec{}.png'.format(filepath,i))
            plt.imshow(og_X[item_index].reshape(256,256,3))
            plt.savefig('{}chosen.png'.format(filepath))
        else:
            return recs

def plot_elbow(X_train_compressed,filename=None):
    ''' Plots the # of k clusters vs. inertia.'''

        distortions = []
        K = range(1,100)
        for k in K:
            kmeans = KMeans(n_clusters=k,max_iter=10, n_jobs=-1)
            kmeans.fit(X_train_compressed)
            distortions.append(kmeans.inertia_)

        # Plot the elbow
        plt.plot(K, distortions)
        plt.grid(True)
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        if filename:
            plt.savefig(filename)

if __name__ == '__main__':
    df = pd.read_csv('s3a://capstone-3/data/subset_df2.csv')
    df.drop('Unnamed: 0',axis=1, inplace=True)

    #for training and testing the autoencoder
    X_train = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/train/*.png')])
    X_train = X_train.reshape(-1, 256, 256, 3)
    X_train = X_train / np.max(X_train)

    X_test = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/test/*.png')])
    X_test = X_test.reshape(-1,256,256,3)
    X_test = X_test/ np.max(X_test)

    X_val = np.array([cv2.imread('{}'.format(file)) for file in glob.glob('data/val/*.png')])
    X_val = X_val.reshape(-1,256,256,3)
    X_val = X_val/ np.max(X_val)

    #for clustering and finding indices of recs
    X_total = [(file,cv2.imread('{}'.format(file))) for file in glob.glob('recolored/*.png')]
    indices_and_arrays = []
    for i in range(len(X_total)):
        split = X_total[i][0].split('/')
        index = int(split[1].split('.')[0])
        indices_and_arrays.append((index,X_total[i][1]))

    indices_and_arrays = sorted(indices_and_arrays)
    X_total_arrays = []
    # indices = []
    for i in range(len(indices_and_arrays)):
        # indices.append(indices_and_arrays[i][0])
        X_total_arrays.append(indices_and_arrays[i][1])
    X_total_arrays = np.array(X_total_arrays)
    X_total_arrays = X_total_arrays.reshape(-1,256,256,3)
    X_total_arrays = X_total_arrays/ np.max(X_total_arrays)

    # use for fitting new autoencoder
    autoencoder = cnn_autoencoder()
    autoencoder.fit(X_train,X_train, epochs=6, validation_data=(X_test, X_test))
    autoencoder.save('models/autoencoder.h5')

    # use to load previous fit autoencoder
    # autoencoder = load_model('models/autoencoder6.h5')

    restored_imgs = autoencoder.predict(X_val)

    indices = np.random.choice(len(restored_imgs),10)
    for i in indices:
        plt.imshow(X_val[i].reshape(256, 256,3))
        plt.savefig('images/restored_color/test{}'.format(i))

        plt.imshow(restored_imgs[i].reshape(256, 256,3))
        plt.savefig('images/restored_color/restored{}'.format(i))

    #Cluster on all of train, val, test images
    one = int(np.floor(len(X_total)/3))
    two = 2 * one
    X_compressed1 = get_compressed_images(autoencoder,X_total_arrays[:one],7)
    X_compressed2 = get_compressed_images(autoencoder,X_total_arrays[one:two],7)
    X_compressed3 = get_compressed_images(autoencoder,X_total_arrays[two:],7)
    X_compressed = np.append(X_compressed1, X_compressed2, axis=0)
    X_compressed = np.append(X_compressed, X_compressed3, axis=0)

    kmeans, labels = cluster_compressed(X_compressed)

    labels_df = pd.DataFrame(labels, columns=['label'])
    images_and_labels = pd.concat([df,labels_df], axis=1)
    images_and_labels.to_csv('images_and_labels.csv')

    item_index = np.random.choice(len(X_compressed))
    recs = get_kmeans_rec(item_index,kmeans,X_total_arrays,5, 'images/rec_test/')
