import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

from keras.datasets import mnist, cifar10, cifar100
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D
from keras.layers.normalization import BatchNormalization

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from skimage import io, util
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from sklearn.utils import Bunch

from trainingSchemes import default_training_scheme
from trainingSchemes import train_on_extracted_context_vectors
from trainingSchemes import train_via_SimplePairing

from emnist_letters_gan import create_augment_and_train_GAN
from emnist_letters_gan import visualise_synthesised_images

from pathlib import Path

np.random.seed(10)


# todo inductive transfer learning using the cifar10 dataset
# todo model depth
# todo test time augmentations


def preprocess(X):
    out_set = []
    for img in X:
        out_set.append(rgb2gray(img))
    return np.asarray(out_set)


# Dr Dane Brown's code modified
def load_image_files(container_path, dimension=(30, 30)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "my dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect', preserve_range=True)

            ################## image pre-processing ##################################################

            # Gaussian blur to remove noise
            img_blurred = gaussian(img_resized, sigma=1, preserve_range=True)
            # Grayscaling
            img_grayed = rgb2gray(img_blurred)
            # Thresholding
            thresh = threshold_otsu(img_grayed.astype('uint32'))
            img_threshed = img_grayed > thresh
            # # Displaying
            # io.imshow(img_threshed)
            # plt.show()

            #####################################

            flat_data.append(img_threshed.flatten())
            images.append(img_threshed)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


def plot_loss_accuracy(H, epochs, label):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epochs), H['loss'], label='Training loss')
    plt.plot(np.arange(0, epochs), H['accuracy'], label='Training accuracy')
    plt.plot(np.arange(0, epochs), H['val_loss'], label='Validation loss')
    plt.plot(np.arange(0, epochs), H['val_accuracy'], label='Validation accuracy')
    title = 'Training and Evaluation accuracy on ' + label
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# def crop_imgs(set=[], labels=[]):
#         """Produces 5N images by taking corner crops and center crops"""
#         print('Cropping images...')
#         out_set = []
#         out_labels = []
        
#         for i in range(len(set)):
#             # centre
#             out_set.append(np.asarray(set[i][1:27, 1:27]))
#             out_labels.append(labels[i])

#             #top left 
#             out_set.append(np.asarray(set[i][:26, :26]))
#             out_labels.append(labels[i])
            
#             # top right
#             out_set.append(np.asarray(set[i][:26, 2:28]))
#             out_labels.append(labels[i])
            
#             # bottom left
#             out_set.append(np.asarray(set[i][2:28, :26]))
#             out_labels.append(labels[i])
            
#             # bottom right
#             out_set.append(np.asarray(set[i][2:28, 2:28]))
#             out_labels.append(labels[i])

#         return np.asarray(out_set), np.asarray(out_labels)

# this should be more general
# test it a lil
def crop_imgs(set=[], labels=[]):
        """Produces 5N images by taking corner crops and center crops"""
        print('Cropping images...')
        out_set = []
        out_labels = []
        height, width = len(set[0]), len(set[0][0])
        y, x = int(height * 0.1), int(width * 0.1)
        

        for i in range(len(set)):
            # centre
            out_set.append(np.asarray(set[i][(0+y//2):(height-(y//2)), (0+x//2):(width-(x//2))]))
            out_labels.append(labels[i])

            #top left 
            out_set.append(np.asarray(set[i][:height-y, :width-x]))
            out_labels.append(labels[i])
            
            # top right
            out_set.append(np.asarray(set[i][:height-y, x:width]))
            out_labels.append(labels[i])
            
            # bottom left
            out_set.append(np.asarray(set[i][y:height, :width-x]))
            out_labels.append(labels[i])
            
            # bottom right
            out_set.append(np.asarray(set[i][y:height, x:width]))
            out_labels.append(labels[i])

        return np.asarray(out_set), np.asarray(out_labels)


def crop_test_imgs(set=[], labels=[]):
        """Produces 5N images by taking corner crops and center crops"""
        print('Cropping images...')
        out_set = []
        out_labels = []
        height, width = len(set[0]), len(set[0][0])
        y, x = int(height * 0.1), int(width * 0.1)
        

        for i in range(len(set)):
            out_set.append(np.asarray(set[i][(0+y//2):(height-(y//2)), (0+x//2):(width-(x//2))]))
            out_labels.append(labels[i])
        return np.asarray(out_set), np.asarray(out_labels)


def load_Chars74K(dim=(48, 48), split=0.5):
    print("Loading Chars74K")
    dir = os.path.dirname(__file__)
    filePath = os.path.join(dir, 'Chars74K/images.npy')
    X_Chars74K, y_Chars74K = [], []
    if(not os.path.isfile(filePath)):
        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'English/Img/GoodImg/Bmp')
        dataset = load_image_files(filePath, dimension=dim)
        X_Chars74K, y_Chars74K = dataset.images, dataset.target

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'English/Img/BadImag/Bmp')
        dataset = load_image_files(filePath, dimension=dim)
        X_Chars74K = np.concatenate((X_Chars74K, dataset.images))
        y_Chars74K = np.concatenate((y_Chars74K, dataset.target))

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'English/Fnt')
        dataset = load_image_files(filePath, dimension=dim)
        X_Chars74K = np.concatenate((X_Chars74K, dataset.images))
        y_Chars74K = np.concatenate((y_Chars74K, dataset.target))

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'English/Hnd/Img')
        dataset = load_image_files(filePath, dimension=dim)
        X_Chars74K = np.concatenate((X_Chars74K, dataset.images))
        y_Chars74K = np.concatenate((y_Chars74K, dataset.target))

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'Chars74K/images.npy')
        np.save(filePath, X_Chars74K)

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'Chars74K/target.npy')
        np.save(filePath, y_Chars74K)
    else:
        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'Chars74K/images.npy')
        X_Chars74K = np.load(filePath)

        dir = os.path.dirname(__file__)
        filePath = os.path.join(dir, 'Chars74K/target.npy')
        y_Chars74K = np.load(filePath)


    return split_data(X_Chars74K, y_Chars74K, split)


def split_data(X, y, split):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
        train_size=split,
        shuffle=True,
    )
    return X_train, y_train, X_test, y_test


#*################## loading datasets ################################

print('(0) NONE; (1) MNIST; (2) EMNIST; (3) cifar10; (4) Chars74K')
print('Select Base Dataset from the options above. Enter anything else to train without a base dataset:')
base_dataset = input()

base_epochs = 0
if base_dataset != '0': 
    print('How long should the model train on the base dataset?')
    base_epochs = int(input())

base_split = 0.7

base_size = 0
base_shape = (0, 0, 0)
base_classes = 0
base_dataset_descr = ''

X_base_train, y_base_train = [], []
X_base_test, y_base_test = [], []


if base_dataset == '1':
    print("Loading MNIST digits dataset...")

    (X_base_train, y_base_train), (X_base_test, y_base_test) = mnist.load_data(path='./mnist.npz')
    X_base_train, X_base_test = X_base_train / 255.0, X_base_test / 255.0
    X_base_train, X_base_test = X_base_train.reshape([60000, 28, 28, 1]), X_base_test.reshape([10000, 28, 28, 1])
    # X = np.concatenate((X_base_train, X_base_test))
    # y = np.concatenate((y_base_train, y_base_test))
    base_size = 28 * 28
    base_shape = (28, 28, 1)
    base_classes = 10
    base_dataset_descr = 'MNIST'

elif base_dataset == '2':
    print("Loading EMNIST leters dataset...")
    dataset = 'letters'

    X_base_train, y_base_train = extract_training_samples(dataset)
    X_base_test, y_base_test = extract_test_samples(dataset)
    # X = np.concatenate((X_train, X_test))
    # y = np.concatenate((y_train, y_test))
    base_size = 28 * 28
    base_shape = (28, 28, 1)
    base_classes = 27
    base_dataset_descr = 'EMNIST'

    # X_base_train, y_base_train, X_base_test, y_base_test = split_data(X, y, base_split)
    X_base_train, X_base_test = X_base_train / 255.0, X_base_test / 255.0
    X_base_train, X_base_test = X_base_train.reshape([X_base_train.shape[0], 28, 28, 1]), X_base_test.reshape([X_base_test.shape[0], 28, 28, 1])

elif base_dataset == '3':
    print("Loading cifar10 dataset...")
    # #* 70 000 images of digits, training set consistis of 60 000 images
    (X_base_train, y_base_train), (X_base_test, y_base_test) = cifar10.load_data()
    X_base_train = preprocess(X_base_train)
    X_base_test = preprocess(X_base_test)
    X_base_train, X_base_test = X_base_train / 255.0, X_base_test / 255.0
    X_base_train, X_base_test = X_base_train.reshape([X_base_train.shape[0], 32, 32, 1]), X_base_test.reshape([X_base_test.shape[0], 32, 32, 1])

    base_size = 32 * 32
    base_shape = (32, 32, 1)
    base_classes = 10
    base_dataset_descr = 'cifar10'

elif base_dataset == '4':    
    base_size = 48 * 48
    base_shape = (48, 48, 1)
    base_classes = 62
    base_dataset_descr = 'Chars74K'
    X_base_train, y_base_train, X_base_test, y_base_test = load_Chars74K(split=base_split)
    
    X_base_train, X_base_test = X_base_train / 255.0, X_base_test / 255.0
    X_base_train, X_base_test = X_base_train.reshape([X_base_train.shape[0], 48, 48, 1]), X_base_test.reshape([X_base_test.shape[0], 48, 48, 1])

# elif base_dataset == '5':
#     print("Loading cifar100 dataset...")
#     (X_base_train, y_base_train), (X_base_test, y_base_test) = cifar100.load_data(label_mode='fine')
#     X_base_train = preprocess(X_base_train)
#     X_base_test = preprocess(X_base_test)
#     X_base_train, X_base_test = X_base_train / 255.0, X_base_test / 255.0
#     X_base_train, X_base_test = X_base_train.reshape([X_base_train.shape[0], 32, 32, 1]), X_base_test.reshape([X_base_test.shape[0], 32, 32, 1])

#     base_size = 32 * 32
#     base_shape = (32, 32, 1)
#     base_classes = 100
#     base_dataset_descr = 'cifar100'

else:
    base_shape = (0,0,0)
print('done')

print('(1) Chars74K; (2) EMNIST')
print('From the above list pick a target dataset')
target_dataest = input()

print('How much of the data should be used for training? Enter a number between 0.0 and 1.0:')
split = float(input())

X_train, y_train = [], []
X_test, y_test = [], []

size = 0
crop_size = 0
shape = (0, 0, 0)
crop_shape = (0, 0, 0)
classes = 0
dataset_descr = ''


if target_dataest == '2':
    print("Loading EMNIST leters dataset...")
    dataset = 'letters'
    X_train, y_train = extract_training_samples(dataset)
    X_test, y_test = extract_test_samples(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    size = 28 * 28
    crop_size = 26 * 26
    shape = (28, 28, 1)
    crop_shape = (26, 26, 1)
    classes = 27
    dataset_descr = 'EMNIST'
    class_layer_size = 1728
    X_train, y_train, X_test, y_test = split_data(X, y, split)

else:
    size = 48 * 48
    crop_size = 44 * 44
    shape = (48, 48, 1)
    crop_shape = (44, 44, 1)
    classes = 62
    dataset_descr = 'Chars74K'
    class_layer_size = 6912

    X_train, y_train, X_test, y_test = load_Chars74K(split=split)
    X_train, X_test = X_train.reshape([X_train.shape[0], 48, 48, 1]), X_test.reshape([X_test.shape[0], 48, 48, 1])
    # X_train, X_test = X_train / 255.0, X_test / 255.0

print('done')

#*################## one-hot-encoding ################################

if base_shape != (0,0,0):
    y_base_train = to_categorical(y_base_train)
    y_base_test = to_categorical(y_base_test)

#*############################ training the base model #############################################

feature_layers = [
    Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=base_shape),
    BatchNormalization(),
    Dropout(0.0),
    
    Conv2D(64, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(96, 3, activation='relu', padding='same'),
    MaxPooling2D(2),
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(96, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(128, 3, activation='relu', padding='same'),
    MaxPooling2D(2),
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.1),
    
    Conv2D(192, 3, activation='relu', padding='same'),
    MaxPooling2D(2),
    BatchNormalization(),
    Dropout(0.1),
]

classification_layers = [
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.0),
]

base_model_name = 'HCR_base_model_' + base_dataset_descr + '_' + str(base_epochs) + 'epochs' + '.pkl'

if base_shape != (0,0,0):
    if(not os.path.isfile(base_model_name)):
        print("Model does not exist... creating and saving model")
        base_model = Sequential(feature_layers + classification_layers + [Dense(base_classes, activation='softmax')])
        base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        base_model.summary()
        epochs = base_epochs
        H = base_model.fit(
            X_base_train,
            y_base_train,
            epochs=epochs,
            verbose=1,
            batch_size=16,
            validation_split=0.2,
            shuffle=True,
            workers=4,
            use_multiprocessing=True
        )
        joblib.dump(base_model, base_model_name)
        plot_loss_accuracy(H.history, epochs, 'base model')
        score = base_model.evaluate(X_base_test, y_base_test)
        print('Accuracy on the base model: ', score[1])
    else:
        print("Model already exists... loading model")
        base_model = joblib.load(base_model_name)
        base_model.summary()

        layers = base_model.layers
        for i in range(24):
            feature_layers[i] = layers[i]
        i = 24
        j = 0
        while j < 3 and i < 27:  
            classification_layers[j] = layers[i]
            i, j = i+1, j+1

#*###################################### training and evaluating the target model #####################################

possible_augmentations = ['(1) extrapolate', '(2) PatchShuffle', '(3) invert images', '(4) swirl images', '(5) morph images', '(6) randomly erase images', '(7) flip images']
print("The possible augmentations are:")
print(possible_augmentations)

augmentations = []
print('Pick one augmentation and press enter.')
print('Repeat until all the augmentations you want are selected.')
print('The augmentations will be performed in the order selected')
while True:
    print("Enter '0' to stop.")
    aug = int(input())
    if aug == 0 or aug > 7:
        break
    augmentations.append(aug-1)


print('For how many epochs should the model train:')
epochs = int(input())

print('How do you want train the model')
print("(1) default; (2) Train on augmented images then original images; (3) SimplePairing; (4) Use cropped images; (5) 'icing on the cake'")
training_scheme = int(input())

if base_size != size:
    classification_layers = [
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.0),
    ]

feature_layers[0] = Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=shape)
target_model = Sequential(feature_layers + classification_layers + [Dense(classes, activation='softmax')])
target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
if training_scheme == 1:
    target_model = default_training_scheme(target_model, X_train, y_train, epochs=epochs, X_test=X_test, y_test=y_test, augmentations=augmentations, size=size, shape=shape, dataset_descr=dataset_descr, classes=classes)

elif training_scheme == 2:
    target_model = default_training_scheme(target_model, X_train, y_train, epochs=epochs, X_test=X_test, y_test=y_test, augmentations=augmentations, size=size, shape=shape, dataset_descr=dataset_descr, classes=classes)

    feature_layers[0] = Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=shape)
    target_model = Sequential(feature_layers + classification_layers + [Dense(classes, activation='softmax')])
    target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    target_model = default_training_scheme(target_model, X_train, y_train, epochs=epochs, X_test=X_test, y_test=y_test, augmentations=[], size=size, shape=shape, dataset_descr=dataset_descr, classes=classes)

elif training_scheme == 3:
    target_model = train_via_SimplePairing(target_model, X_train, y_train, X_test=X_test, y_test=y_test, augmentations=augmentations, size=size, shape=shape, dataset_descr=dataset_descr, classes=classes)

elif training_scheme == 4:
    feature_layers[0] = Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=crop_shape)
    target_model = Sequential(feature_layers + classification_layers + [Dense(classes, activation='softmax')])
    target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    target_model.summary()
    
    X_train, y_train = crop_imgs(set=X_train, labels=y_train)
    X_test, y_test = crop_test_imgs(set=X_test, labels=y_test)

    target_model = default_training_scheme(target_model, X_train, y_train, epochs=epochs, X_test=X_test, y_test=y_test, augmentations=augmentations, size=crop_size, shape=crop_shape, dataset_descr=dataset_descr, classes=classes)
elif training_scheme == 5:
    target_model.summary()
    classifier = train_on_extracted_context_vectors(
        target_model, 
        X_train, y_train,
        X_test=X_test, y_test=y_test,
        epochs=epochs,
        size=size, 
        shape=shape, 
        augmentations=augmentations,
        layers=25,
        classes=classes,
        dataset_descr=dataset_descr,
        class_layer_size=class_layer_size
    )