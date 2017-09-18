import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import misc

tf.logging.set_verbosity(tf.logging.INFO)

def loadImageData():
    # Make a queue of file names including all the PNG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./train_small/*.png"))

    # Read an entire image file which is required since they're PNGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_png(image_file)

    # print(image.shape)

    return image

def loadTrainCSVData():

    labelMappingDict = {'airplane' : 0, 'automobile' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9}

    df = pd.read_csv("./trainLabels.csv")

    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    label_encoder = lambda l: labelMappingDict[l]
    labels = np.fromiter((label_encoder(label) for label in labels), np.int32)
    labels = tf.one_hot(labels, 10)

    print("labels: %s" % (labels[1]))

    return (features, labels)

def cnn_model_fn(_features, _labels, _mode):

    # input layer
    input_layer = tf.reshape()

    return

def main(_):
    # # Start new session
    with tf.Session() as sess:
        # Required to get the filename matching to run
        tf.global_variables_initializer().run()
        images = loadImageData()
        features, labels = loadTrainCSVData()
        # sess.run(labels[1])

    # images = loadImageData()
    # features, labels = loadTrainCSVData()
    



if __name__ == "__main__":
    tf.app.run()