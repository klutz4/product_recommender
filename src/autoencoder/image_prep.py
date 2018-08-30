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

##Use these functions for use with image urls
def show_image_from_url(item_index):
    response = requests.get(products.image_url.iloc[item_index])
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.show()

def convert_image_to_matrix(item_index):
    try:
        response = requests.get(products.image_url.iloc[item_index])
        img = Image.open(BytesIO(response.content))
        arr = np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 3)
    except:
        arr = np.NaN
    return arr

def get_image_arrays_from_url(df):
    image_arrays = []
    for idx in range(0,len(df)): #change to len(products) once on AWS
        image_arrays.append(convert_image_to_matrix(idx))
    df = pd.DataFrame(np.array(image_arrays), columns = ['image_array'])
    df.dropna(inplace=True)
    return df

def save_images(df):
    for idx in range(len(df)):
        try:
            urllib.request.urlretrieve(df.image_url.iloc[idx],'images/{}'.format(idx))
        except:
            continue

##Use these functions for use with saved images
def show_image_from_file(filename):
    img = Image.open(filename)
    plt.imshow(img)
    plt.show()

def resize_and_save_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256,256))
    cv2.imwrite('/data/resized/{}'.format(filename), img)


# DO NOT NEED ??
# def get_image_arrays_from_file(filepath):
#     image_arrays = []
#     images = glob.glob(filepath)
#     for file in images:
#         image_arrays.append(load_and_resize_image(file))
#     df = pd.DataFrame(np.array(image_arrays), columns = ['image_array'])
#     df.dropna(inplace=True)
#     return df

## Use this function to save images from image urls
def save_images_to_local(df):
    ''' Save images from image url.'''
    for item_index in range(len(df)):
        try:
            response = requests.get(df.image_url.iloc[item_index])
            img = Image.open(BytesIO(response.content))
            plt.imshow(img)
            plt.savefig('data/og_images/{}.png'.format(item_index))
        except:
            continue

## Use these functions to send images to s3 bucket
def save_to_s3(filename):
    s3 = boto3.resource('s3')
    s3.Bucket('capstone-3').upload_file(filename, 'images/{}'.format(filename))

def save_folder_to_s3(folder_path):
    images = glob.glob(folder_path)
    for file in images:
        save_to_s3(file)

#Split images into train, val and test
def split_images(filepath):
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
    for file in filenames:
        img = cv2.imread(file)
        cv2.imwrite('../{}/{}'.format(directory,file),img)

if __name__ == '__main__':

    pd.set_option('display.max_columns', 500)
    products = pd.read_csv('s3a://capstone-3/data/products_w_images.csv')
    products.drop('Unnamed: 0',axis=1, inplace=True)
    products = products[products['category'] == 'art']
    products = products.reset_index(drop=True)

    # Prep 1060 images from total df of ~16,000
    save_images_to_local(products.iloc[:1060])

    images = glob.glob('data/og_images/*.png')
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
