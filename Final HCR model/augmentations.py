import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from skimage.morphology.grey import opening

from sklearn.neighbors import KNeighborsClassifier

from skimage import io, util
from skimage.morphology import dilation, closing, erosion, closing
from skimage.transform import swirl

from random import sample


# todo cropping
# todo random erasing
# todo SimplePairing
# todo Icing on the Cake
# todo extrapolation
# todo interpolaation
# todo PatchShuffle

class AugmentImages():
    # todo determine best no. of neighbours
    knn_classifier = KNeighborsClassifier(n_neighbors=15)
    size = 0
    shape = (0,0,0)
    dataset_descr = ''

    def __init__(self, set, labels, size=28*28, shape=(28,28,1), dataset_descr='EMNIST'):
        print('finding k-Nearest Neighbours')
        # todo need to handle input shapes better
        temp_set = set.reshape([set.shape[0], size])
        self.knn_classifier.fit(temp_set, labels)
        self.base_set = set
        self.size = size
        self.shape = shape
        self.dataset_descr = dataset_descr

    def find_kNN(self, img, k=11):
        img = img.reshape(1, -1)
        neighbours = self.knn_classifier.kneighbors(X=img, n_neighbors=k, return_distance=False)
        return neighbours


    # todo need to make better use of the reconstructor
    # def extrapolate(set, labels, reconstructor=Sequential()):
    def extrapolate(self, set=[], labels=[]):
        """Extrapolates each image with its 10 nearest neighbours"""
        print('Extrapolating images')
        # todo take in the number of samples to make reshaping easier
        # todo optionally take in the shape of data for displaying
        out_set = []
        out_labels = []

        for i in range(len(set)):
            vector = set[i]
            label = labels[i]
            out_set.append(vector)
            out_labels.append(label)

            # width = vector.shape[0]
            # height = vector.shape[1]
            # channels = vector.shape[2]
           
            neighbours = self.find_kNN(vector.reshape([self.size]))[0, 1:]
            for idx in neighbours:
                extrapolated_vector = ((vector - self.base_set[idx]) * 0.5) + vector
                out_set.append(extrapolated_vector)
                out_labels.append(label)

        return np.asarray(out_set), np.asarray(out_labels)

    
    # doesn't work on Chars74K
    def PatchShuffle(self, set=[], labels=[], ps=2):
        """Performs PatchShuffle on 4x4 patches"""
        # todo select between 2x2 and 4x4
        print("Performing PatchShuffle...")
        set = set.reshape([set.shape[0], self.shape[0], self.shape[1], self.shape[2]])
        out_set = []
        out_labels = []

        for i in range(len(set)):
            img = set[i].copy()

            temp_img = copy.deepcopy(img)
            label = labels[i]

            out_set.append(img)
            out_labels.append(label)
            
            x = 1
            y = 1
            # Randomly rearanges pixels in a 4x4 patch 
            
            # !tessssssssssssssssssssssssssssssssss

            # if img.shape[0] % ps != 0 or img.shape[0] % ps != 0:
            #     print("incompatible patch shuffle shape: ",  ps, "x", ps, " img shape: ", img.shape)

            # i = ps - 1
            # j = ps - 1
            # while i < img.shape[0] and j < img.shape[1]:
            #     tempi = i
            #     tempj = j
            #     while tempi >= 0:
            #         while tempj >= 0:
                        
            #             shuffled = sample([temp_img[x][y], temp_img[x-1][y], temp_img[x][y-1], temp_img[x-1][y-1]], 4)
            #             temp_img[x][y], temp_img[x-1][y], temp_img[x][y-1], temp_img[x-1][y-1] = shuffled[0], shuffled[1], shuffled[2], shuffled[3] 
                        
            #     i = i + ps
            #     j = j + ps

            # !doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee


            while x < img.shape[0]:
                y = 1
                while y < img.shape[1]:
                    shuffled = 0
                    shuffled = sample([temp_img[x][y], temp_img[x-1][y], temp_img[x][y-1], temp_img[x-1][y-1]], 4)
                    temp_img[x][y], temp_img[x-1][y], temp_img[x][y-1], temp_img[x-1][y-1] = shuffled[0], shuffled[1], shuffled[2], shuffled[3] 
                    
                    shuffled = sample([temp_img[x][y-2], temp_img[x-2][y-2], temp_img[x][y-3], temp_img[x-1][y-3]], 4)
                    temp_img[x][y-2], temp_img[x-2][y-2], temp_img[x][y-3], temp_img[x-1][y-3] = shuffled[0], shuffled[1], shuffled[2], shuffled[3]

                    shuffled = sample([temp_img[x-2][y], temp_img[x-3][y], temp_img[x-2][y-1], temp_img[x-3][y-1]], 4)
                    temp_img[x-2][y], temp_img[x-3][y], temp_img[x-2][y-1], temp_img[x-3][y-1] = shuffled[0], shuffled[1], shuffled[2], shuffled[3] 
                    
                    shuffled = sample([temp_img[x-2][y-2], temp_img[x-3][y-2], temp_img[x-2][y-3], temp_img[x-3][y-3]], 4)
                    temp_img[x-2][y-2], temp_img[x-3][y-2], temp_img[x-2][y-3], temp_img[x-3][y-3] = shuffled[0], shuffled[1], shuffled[2], shuffled[3] 

                    y = y + 2
                x = x + 2

            out_set.append(temp_img)
            out_labels.append(label)
        return np.asarray(out_set), np.asarray(out_labels)


    def invert_imgs(self, set=[], labels=[]):
        """Inverts each image"""
        print('Inverting images...')
        out_set = []
        out_labels = []
        for i in range(0, len(set)):
            out_set.append(set[i])
            out_labels.append(labels[i])

            inverted_img = util.invert(set[i])
            out_set.append(inverted_img)
            out_labels.append(labels[i])

        return np.asarray(out_set), np.asarray(out_labels)

    
    def crop_imgs(set=[], labels=[]):
        """Produces 5N images by taking corner crops and center crops"""
        print('Cropping images...')
        out_set = []
        out_labels = []
        
        for i in range(len(set)):
            # centre
            out_set.append(np.asarray(set[i][1:27, 1:27]))
            out_labels.append(labels[i])
            
            #top left 
            out_set.append(np.asarray(set[i][:26, :26]))
            out_labels.append(labels[i])
            
            # top right
            out_set.append(np.asarray(set[i][:26, 2:28]))
            out_labels.append(labels[i])
            
            # bottom left
            out_set.append(np.asarray(set[i][2:28, :26]))
            out_labels.append(labels[i])
            
            # bottom right
            out_set.append(np.asarray(set[i][2:28, 2:28]))
            out_labels.append(labels[i])

        return np.asarray(out_set), np.asarray(out_labels)


    def swirl_imgs(self, set=[], labels=[]):
        """swirls each image 3 times with a random strength between 0.5 and 2.0"""
        print('Swirling images...')
        out_set = []
        out_labels = []
        for i in range(0, len(set)):
            out_set.append(set[i])
            out_labels.append(labels[i])

            j = 0
            while j < 3:
                j += 1
                str = np.random.uniform(0.5, 2.0)
                img_swirled = swirl(set[i], strength=str)
                out_set.append(img_swirled)
                out_labels.append(labels[i])
        return np.asarray(out_set), np.asarray(out_labels)


    def morph_imgs(self, set=[], labels=[]):
        """Adds dilated and closed Images"""
        print('Morphing images...')
        out_set = []
        out_labels = []
        for i in range(0, len(set)):
            out_set.append(set[i])
            out_labels.append(labels[i])

            img_dilated = dilation(set[i])
            out_set.append(img_dilated)
            out_labels.append(labels[i])

            img_closed = closing(set[i])
            out_set.append(img_closed)
            out_labels.append(labels[i])

            img_eroded = erosion(set[i])
            out_set.append(img_eroded)
            out_labels.append(labels[i])

            img_opened = opening(set[i])
            out_set.append(img_opened)
            out_labels.append(labels[i])
        return np.asarray(out_set), np.asarray(out_labels)


    def random_erase_imgs(self, set=[], labels=[]):
        """Randomly fills a square patch with white, black and random values"""
        print("Randomly erasing images...")
        out_set = []
        out_labels = []
        for i in range(0, len(set)):
            out_set.append(set[i])
            out_labels.append(labels[i])

            img_width, img_height = set[i].shape[0], set[i].shape[1]
            start_x, start_y = np.random.randint(0, img_width - 5), np.random.randint(0, img_height - 5)

            eraser_width = np.random.randint(5, img_width - start_x)
            eraser_height = np.random.randint(5, img_height - start_y)

            temp_img_white = set[i] * 1.0
            temp_img_black = set[i] * 1.0
            temp_img_rand = set[i] * 1.0
            for row in range(start_y, start_y + eraser_height):
                for column in range(start_x, start_x + eraser_width):
                    temp_img_white[row, column] = 255
                    temp_img_black[row, column] = 0
                    temp_img_rand[row, column] = np.random.randint(0, 256)

            out_set.append(temp_img_white)
            out_labels.append(labels[i])

            out_set.append(temp_img_black)
            out_labels.append(labels[i])

            out_set.append(temp_img_rand)
            out_labels.append(labels[i])
        return np.asarray(out_set), np.asarray(out_labels)


    # for emnist letters dataset
    def flip_imgs(self, set=[], labels=[]):
        """flips 'flippable' images (Currently only works for the emnist dataset)"""
        print('Flipping images...')
        out_set = []
        out_labels = []
        
        horizontal_flip_set = []
        vertical_flip_set = []
        diagonal_flip_set = []
        horizontal_and_vertical_and_diagonal_flip_set = []

        if self.dataset_descr == 'EMNIST':
            # These arrays contain the classes which can be 
            # flipped horizontally, vertically,
            # horizontally or vertically,
            # and flipped horizantally and vertically
            # !double check if the emnist transforms are correct, these are applied generally, it might be better to have unique flips that handle the uppercase and lowercase cases better
            horizontal_flip_set = [1, 13, 20, 21, 22, 23, 25]
            vertical_flip_set = [2, 3, 4, 5, 11]
            diagonal_flip_set = [19, 26]
            horizontal_and_vertical_and_diagonal_flip_set = [8, 9, 15, 24] 
        
        elif self.dataset_descr == 'Chars74K':
            horizontal_flip_set = [11, 23, 30, 31, 32, 33, 35, 49, 50, 57, 58, 59, 61]
            vertical_flip_set = [3, 12, 13, 14, 15, 21, 39]
            diagonal_flip_set = [29, 36, 55, 62]
            horizontal_and_vertical_and_diagonal_flip_set = [0, 1, 8, 18, 19, 25, 34, 45, 48, 51, 60]

        for i in range(0, len(set)):
            out_set.append(set[i])
            out_labels.append(labels[i])

            if horizontal_flip_set.__contains__(labels[i]):
                img_flipped = np.fliplr(set[i])
                out_set.append(img_flipped)
                out_labels.append(labels[i])

            if vertical_flip_set.__contains__(labels[i]):
                img_flipped = np.flipud(set[i])
                out_set.append(img_flipped)
                out_labels.append(labels[i])

            if horizontal_and_vertical_and_diagonal_flip_set.__contains__(labels[i]):
                img_flipped_hor = np.fliplr(set[i])
                out_set.append(img_flipped_hor)
                out_labels.append(labels[i])

                img_flipped_vert = np.flipud(set[i])
                out_set.append(img_flipped_vert)
                out_labels.append(labels[i])
                
                img_flipped_diag = np.flipud(img_flipped_hor)
                out_set.append(img_flipped_diag)
                out_labels.append(labels[i])

            if diagonal_flip_set.__contains__(labels[i]):
                img_flipped_vert = np.flipud(set[i])
                img_flipped_diag = np.fliplr(img_flipped_vert)
                out_set.append(img_flipped_diag)
                out_labels.append(labels[i])
        return np.asarray(out_set), np.asarray(out_labels)