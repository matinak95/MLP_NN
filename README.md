The code implements a multi layer perceptron(MLP) feedforward/backpropagation
neural network from the scratch and uses it to classify the
hand-written digits dataset, MNIST.

* Note: The code only benefits from numerical libraries such as Numpy, but
no machine learning libraries has been used for this implementation.

This dataset consists of four files:
1. train-images-idx3-ubyte.gz , which contains
60,000 28x28 grayscale training images,
each representing a single handwritten digit.

2. train-labels-idx1-ubyte.gz , which contains
the associated 60,000 labels for the training images.

3. t10k-images-idx3-ubyte.gz , which contains 10,000
28x28 grayscale testing images, each representing a single handwritten digit.

4. t10k-labels-idx1-ubyte.gz , which contains
the associated 10,000 labels for the testing images.

Each of the 28x28 = 784 pixels of each of these images
is represented by a single 8-bit color channel.
Thus, the values each pixel can take on
range from 0 (completely black) to 255 (28 − 1, completely white).

We converted the pickled dataset to .csv format.

The code gets 3 input files:
1. train_image.csv
2. train_label.csv
3. test_image.csv

It outputs: test_predictions.csv.

* For performing backpropagation, Mean Squared Error (MSE) loss function
and Stochastic Gradient Descent (SGD) have been utilized.

The hyper-parameters are:
1. Number of hidden layers
2. Number of cells in the hidden layer(s)
3. Learning rate
4. Number of epochs for training
5. Batch size
