import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import shutil
import os
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')

#%matplotlib inline
seed = 7
np.random.seed(seed)


df = pd.read_csv('leafsnap-dataset/leafsnap-dataset-images.txt', sep='\t')
image_size=224
df['genus'] = df['species'].str.split(expand=True)[0]
leafsnap = 'leafsnap-dataset'

i = 1
for image in df[df.source=='field'].image_path:
    img = cv2.imread("%s/%s" % (leafsnap, image))
    plt.figure()
    plt.imshow(img)
    i = i-1;
    if(i == 0):
        break


df.groupby('source').count()

df.head()
# get one image per Ulmus (Elm) tree type
df3 = df[df.genus=='Ulmus'].groupby('species').min()
for name, path in zip(df3.axes[0], df3['image_path']):
    print(name, path)
    img = cv2.imread("%s/%s" % (leafsnap, path))
    plt.figure()
    plt.imshow(img)


train_dir = 'data/train'
test_dir = 'data/test'
validation_dir = 'data/validation'

if os.path.exists('data'):
    shutil.rmtree('data')
os.mkdir('data')


def resize_padded(img, new_shape, fill_cval=None, order=1):
    import numpy as np
    fill_cval = img[1,1] # np.min(img)
    if img.shape[0] > img.shape[1]:
        ratio = img.shape[0] / new_shape[0]
    else:
        ratio = img.shape[1] / new_shape[1]
    img2 = cv2.resize(img, (0,0), fx = 1/ratio, fy = 1/ratio)
    new_img = np.empty(new_shape)
    new_img.fill(fill_cval)
    new_img[0:img2.shape[0],0:img2.shape[1]] = img2        
    return(new_img)

def addSampleToDirectory(sourcedir, outdir, samples):
    if not os.path.exists(outdir): os.makedirs(outdir)
    for index, item in samples.iterrows():
        fn = "%s/%s" % (sourcedir, item['image_path'])
        gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        
        name = os.path.basename(fn)
        outfn = "%s/%s" % (outdir, name)
        
        # crop color bars in 'lab' images
        if item['source'] == 'lab':
            h=gray.shape[1]-160
            w=gray.shape[0]-100
            gray = gray[0:w, 0:h]

        img3 = resize_padded(gray, (image_size,image_size))
        img4 = (img3 / np.max(img3) * 255).astype(int)
        cv2.imwrite(outfn, img4)


#species = df.groupby('species')
for sp, data in species:
    # get 80% to train, 10% for validation, 10% for test
    name = sp.lower().replace(" ","_")
    train=data.sample(frac=0.8)
    validation=data.drop(train.index)
    test=validation.sample(frac=0.5)
    validation=validation.drop(test.index)
    addSampleToDirectory("leafsnap-dataset", "%s/%s" % (train_dir, name), train)
    addSampleToDirectory("leafsnap-dataset", "%s/%s" % (validation_dir, name), validation)
    addSampleToDirectory("leafsnap-dataset", "%s/%s" % (test_dir, name), test)


gray = cv2.imread('leafsnap-dataset/dataset/images/field/ulmus_americana/1249059004_0000.jpg', cv2.IMREAD_GRAYSCALE)

gray.shape

batch_size = 32

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
#         horizontal_flip=True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')


