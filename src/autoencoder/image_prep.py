import matplotlib
matplotlib.use('Agg') #for AWS
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import urllib.request
import matplotlib.pyplot as plt
import boto3
import glob
import cv2
import random

## Use this function to save images from image urls
def save_images_to_local(subset):
    ''' Save images from image url.
    Input:
    subset = subset of original dataset of images to save

    Output:
    images saved to local machine
    '''
    for item_index in range(subset.index[0],(subset.index[0]+len(subset.index))):
        try:
            response = requests.get(products.image_url.iloc[item_index])
            img = Image.open(BytesIO(response.content))
            plt.imshow(img)
            plt.savefig('/data/og_images2/{}.png'.format(item_index))
        except:
            continue

##Use these functions for use with saved images
def show_image_from_file(filename):
    '''Show one image.'''
    img = Image.open(filename)
    plt.imshow(img)
    plt.show()

def resize_and_save_image(filename):
    '''Resize image to 256x256 and save.'''
    img = cv2.imread(filename)
    img = cv2.resize(img, (256,256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/home/ubuntu/product_recommender/product_recommender/data/recolored/{}'.format(filename), img)

def save_to_s3(filename):
    '''Send images to s3 bucket.'''
    s3 = boto3.resource('s3')
    s3.Bucket('capstone-3').upload_file(filename, 'images/{}'.format(filename))

def save_folder_to_s3(folder_path):
    images = glob.glob(folder_path)
    for file in images:
        save_to_s3(file)

def split_images(filepath):
    '''Split images in to train, test, val for autoencoder training.
    Input:
    filepath = where all images are

    Output:
    train_filenames = filenames to go into the train set, list of strings
    test_filenames= filenames to go into the test set, list of strings
    val_filenames = filenames to go into the val set, list of strings
    '''
    filenames = glob.glob(filepath)
    filenames.sort()
    random.shuffle(filenames)

    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    train_filenames = filenames[:split_1]
    val_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]
    return train_filenames, val_filenames, test_filenames

def save_files_after_split(filenames,directory):
    '''Save corresponding files to the corresponding folder (train, test or val).
    Input:
    filenames = list of strings of filenames
    directory = folder to save files to

    Output:
    None
    '''
    for file in filenames:
        img = cv2.imread(file)
        cv2.imwrite('../{}/{}'.format(directory,file),img)

if __name__ == '__main__':

    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('s3a://capstone-3/data/art_only_images.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)

    # Prep subset of images from total df of ~16,000
    subset = products.iloc[:2000]
    save_images_to_local(subset)

    images = glob.glob('*.png')
    for file in images:
        resize_and_save_image(file)

    save_folder_to_s3('resized/*.png')

    train_filenames, val_filenames, test_filenames = split_images('*.png')
    save_files_after_split(train_filenames,'train')
    save_files_after_split(test_filenames,'test')
    save_files_after_split(val_filenames,'val')

    save_folder_to_s3('train/*.png')
    save_folder_to_s3('test/*.png')
    save_folder_to_s3('val/*.png')
