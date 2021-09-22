import sys
import csv
import numpy as np


def CSVReader(filename):
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            rows.append(row)

        line_num = csvreader.line_num

        return rows, line_num


def CSVWriter(filename, excel):
    with open(filename, 'w') as out_file:
        wr = csv.writer(out_file, 'excel')

        for each in excel:
            wr.writerow([str(each)])


class LearnerMain:
    def __init__(self):

        self.train_images, self.trim_num = CSVReader("train_image.csv")
        self.train_labels, self.trl_num = CSVReader("train_label.csv")

        self.test_images, self.test_images_num = CSVReader("test_image.csv")

        self.image_size = 784
        self.batch_size = 15  # 15
        self.output_size = 10
        self.left_layer_num = 50  # 50
        if self.trim_num < 20000:
            self.epoch_number = 55
        elif self.trim_num < 30000:
            self.epoch_number = 45
        else:
            self.epoch_number = 30

        self.learning_rate = 3.3  # 4

        self.current_image_batch = np.array(self.train_images[0: self.batch_size], dtype=float)
        self.batch_number = int(self.trim_num / self.batch_size)
        self.current_labels_batch = np.array(self.train_labels[0: self.batch_size], dtype=float)

        self.wl = np.random.randn(self.left_layer_num, self.image_size)
        self.bl = np.random.randn(self.left_layer_num, 1)

        self.wo = np.random.randn(self.output_size, self.left_layer_num)
        self.bo = np.random.randn(self.output_size, 1)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def yMaker(self, label):
        vector = np.zeros((10, 1))
        vector[int(label[0])] = 1
        return vector

    def updater(self):

        harp_bl = np.zeros((self.left_layer_num, 1), dtype=float)
        harp_wl = np.zeros((self.left_layer_num, self.image_size), dtype=float)
        harp_bo = np.zeros((self.output_size, 1), dtype=float)
        harp_wo = np.zeros((self.output_size, self.left_layer_num), dtype=float)

        for image, label in zip(self.current_image_batch, self.current_labels_batch):
            image = np.reshape(image, (784, 1))
            image = image / 100
            label = self.yMaker(label)

            gradient_wl, gradient_wo, gradient_bl, gradient_bo = self.backPropagation(image, label)

            harp_bo = harp_bo + gradient_bo
            harp_wo = harp_wo + gradient_wo
            harp_bl = harp_bl + gradient_bl
            harp_wl = harp_wl + gradient_wl

        self.wl = self.wl - (self.learning_rate / self.batch_size) * harp_wl
        self.wo = self.wo - (self.learning_rate / self.batch_size) * harp_wo
        self.bl = self.bl - (self.learning_rate / self.batch_size) * harp_bl
        self.bo = self.bo - (self.learning_rate / self.batch_size) * harp_bo

    def backPropagation(self, image, label):

        zo, output, zl, output_left = self.feedForward(image)

        derive = (output - label) * self.sigmoid_deriv(zo)  # 1 * 10
        grad_bo = derive  # 10*1
        grad_wo = np.dot(derive, output_left.transpose())

        derive = np.dot(self.wo.transpose(), derive) * self.sigmoid_deriv(zl)  # left_layer_number*1
        grad_bl = derive  # left_layer_num*1
        grad_wl = np.dot(derive, image.transpose())  # left_layer_number* image_size

        return grad_wl, grad_wo, grad_bl, grad_bo

    def feedForward(self, image):

        wl = self.wl  # image_size*left_layer_number
        wxl = np.dot(wl, image)  # left_layer_num*1
        bl = self.bl  # left_layer_number*1
        zl = wxl + bl
        output_left = self.sigmoid(zl)  # 30*1

        wo = self.wo  # left_layer_num * Output_size
        wxo = np.dot(wo, output_left)
        bo = self.bo  # output_size*1
        zo = wxo + bo  # batch * output_size
        output = self.sigmoid(zo)

        return zo, output, zl, output_left

    def learn(self):

        for epoch in range(self.epoch_number):

            for batch in range(self.batch_number):
                self.current_image_batch = \
                    np.array(self.train_images[batch * self.batch_size:(batch + 1) * self.batch_size], dtype=float)

                self.current_labels_batch = \
                    np.array(self.train_labels[batch * self.batch_size:(batch + 1) * self.batch_size], dtype=float)
                self.updater()

        excel = []
        for image in zip(self.test_images):
            image = np.reshape(image, (784, 1)).astype(float)
            image = image / 100

            zo, output, zl, output_left = self.feedForward(image)

            excel.append(np.argmax(output))

        return excel


if __name__ == "__main__":
    learner = LearnerMain()
    output_file = learner.learn()
    CSVWriter("test_predictions.csv", output_file)
