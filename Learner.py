import numpy as np
import sys
from csv_reader import CSVReader
import random
import copy
import matplotlib.pyplot as plt
import math




class Learner:
    def __init__(self):
        # arr1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        # arr2 = arr2.transpose()
        # print(np.dot(arr1, arr2))
        # sys.exit(0)

        self.iter = 0

        self.image_size = 784
        self.batch_size = 3
        self.output_size = 10
        self.left_layer_num = 10 * self.batch_size
        self.right_layer_num = 5 * self.batch_size
        self.epoch_number = 10

        self.train_images, self.trim_num = CSVReader("train_image.csv")
        self.train_labels, self.trl_num = CSVReader("train_label.csv")
        self.tl_probs = np.zeros((self.batch_size, self.output_size), dtype=float)
        self.current_image_batch = np.array(self.train_images[0: self.batch_size], dtype=float)
        self.batch_number = int(self.trim_num / self.batch_size)
        self.current_labels_batch = np.array(self.train_labels[0: self.batch_size], dtype=float)
        self.yTrainLabels = np.zeros((self.batch_size, self.output_size), dtype=float)

        self.wl = np.zeros((self.image_size, self.left_layer_num), dtype=float)
        # wl[i][j] stands for i: input pixel number j: left cell number
        self.bl = np.zeros((self.batch_size, self.left_layer_num), dtype=float)
        self.zl = np.zeros((self.batch_size, self.left_layer_num), dtype=float)
        self.output_left = np.zeros((self.batch_size, self.left_layer_num), dtype=float)

        self.wr = np.zeros((self.left_layer_num, self.right_layer_num), dtype=float)
        # wr[i][j] stands for i: left cell number j: right cell number
        self.br = np.zeros((self.batch_size, self.right_layer_num), dtype=float)
        self.zr = np.zeros((self.batch_size, self.right_layer_num), dtype=float)
        self.output_right = np.zeros((self.batch_size, self.right_layer_num), dtype=float)

        self.wo = np.zeros((self.right_layer_num, self.output_size), dtype=float)
        self.bo = np.zeros((self.batch_size, self.output_size), dtype=float)
        self.zo = np.zeros((self.batch_size, self.output_size), dtype=float)
        self.output = np.zeros((self.batch_size, self.output_size), dtype=float)

        self.cost_output = np.zeros((self.batch_size, self.output_size), dtype=float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def cost_finder(self):
        print(((self.yTrainLabels - self.output)**2).mean(axis=1))
        sys.exit(0)

    def yMaker(self):
        for index in range(self.batch_size):
            self.yTrainLabels[index][int(self.current_labels_batch[index])] = 1
        with np.printoptions(threshold=np.inf):
            print("yTrainLabels")
            print(self.yTrainLabels)

    def feedForward(self):

        current_image_batch = self.current_image_batch

        wl = self.wl
        wxl = np.dot(current_image_batch, wl)
        bl = self.bl
        self.zl = wxl + bl
        self.output_left = self.sigmoid(self.zl)

        wr = self.wr
        wxr = np.dot(self.output_left, wr)
        br = self.br
        self.zr = wxr + br
        self.output_right = self.sigmoid(self.zr)

        wo = self.wo
        wxo = np.dot(self.output_right, wo)
        bo = self.bo
        self.zo = wxo + bo
        self.output = self.softmax(self.zo)

    def backPropagation(self):

        return

    def learn(self):


        for epoch in range(self.epoch_number):
            for batch in range(self.batch_number):
                self.current_image_batch = \
                    np.array(self.train_images[batch * self.batch_size:(batch + 1) * self.batch_size], dtype=float)

                self.current_labels_batch = \
                    np.array(self.train_labels[batch * self.batch_size:(batch + 1) * self.batch_size], dtype=float)

                self.yMaker()

                with np.printoptions(threshold=np.inf):
                    print("current label batch")
                    print(self.current_labels_batch)

                self.cost_finder()

                self.feedForward()


                sys.exit(0)

        # with np.printoptions(threshold=np.inf):
        #     print(wr)
